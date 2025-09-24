from datetime import datetime

from Agent import Agent, Action, get_client
import random
import requests
import os
import numpy as np

from openai import OpenAI


class Post:
    def __init__(
        self,
        post_id: int,
        author: Agent,
        timestamp: datetime,
        content: str,
        show_info: bool = True,
        calculate_bridging: bool = False,
    ):
        self.post_id = post_id
        self.author = author
        self.timestamp = timestamp
        self.content = content

        self.reposts = 0
        self.reposters = []

        self.show_info = show_info

        if calculate_bridging:
            self.bridging_score = self._calculate_bridging_score()

    def __str__(self):
        post_string = f"""Post ID: {self.post_id}"""

        if self.show_info:
            post_string += f"""
Posted by: user with {self.author.followers} followers
Reposts: {self.reposts}"""

        post_string += f"""
Content: {self.content}"""

        return post_string

    def __repr__(self):
        return f"User {self.author} posted: {self.content}"

    def _calculate_bridging_score(self, retries: int = 3) -> float:
        if retries > 3:
            print("Retries exceeded")
            return 0.0

        body_json = {
            "comment": {"text": self.content},
            "languages": ["en"],
            "requestedAttributes": {
                "AFFINITY_EXPERIMENTAL": {},
                "COMPASSION_EXPERIMENTAL": {},
                "CURIOSITY_EXPERIMENTAL": {},
                "NUANCE_EXPERIMENTAL": {},
                "PERSONAL_STORY_EXPERIMENTAL": {},
                "REASONING_EXPERIMENTAL": {},
                "RESPECT_EXPERIMENTAL": {},
            },
        }

        try:
            r = requests.post(
                f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={os.environ.get('PERSPECTIVE_API_KEY')}",
                json=body_json,
            )
            perspective_response = r.json()

            attributes = [
                "AFFINITY_EXPERIMENTAL",
                "COMPASSION_EXPERIMENTAL",
                "CURIOSITY_EXPERIMENTAL",
                "NUANCE_EXPERIMENTAL",
                "PERSONAL_STORY_EXPERIMENTAL",
                "REASONING_EXPERIMENTAL",
                "RESPECT_EXPERIMENTAL",
            ]

            scores = []
            for attribute in attributes:
                attribute_score = perspective_response["attributeScores"][attribute][
                    "summaryScore"
                ]["value"]
                scores.append(attribute_score)
        except:
            print("Error in perspective API call")
            return self._calculate_bridging_score(retries=retries + 1)

        return sum(scores) / len(scores)

    def json(self):
        """
        Return the post as a JSON object for logging purposes.
        """
        return {
            "post_id": self.post_id,
            "author": self.author.identifier,
            "timestamp": self.timestamp,
            "content": self.content,
            "reposts": self.reposts,
            "reposters": self.reposters,
        }

    def count_repost(self, reposter_id: int):
        """
        Register a repost of the post by a user.
        """
        self.reposters.append(reposter_id)
        self.reposts += 1

    def reposted_by(self, reposter_id: int):
        """
        Returns True if the post has been reposted by the user, else False.
        """
        return reposter_id in self.reposters


class Platform:
    def __init__(
        self,
        user_link_strategy: str = "on_repost",
        timeline_select_strategy: str = "random",
        show_info: bool = True,
    ):
        # All users on the platform
        self.users: list[Agent] = []

        # Of the form {"user_id": int, "post_id": int, "post_content": Post}
        # All posts, including reposts
        self.posts: list[dict] = []

        # Only posts written by users
        self.raw_posts: list[Post] = []

        # Of the form (user_id_link_from, user_id_link_to)
        self.user_links: list[(int, int)] = []

        # Register all actions on the platform
        self.actions: list[dict] = []

        # Keep track of network after each iteration for analysis
        self.network_snapshots = []

        # User link strategy: when to link users
        # on_repost: link users when one user reposts another user's post
        # on_repost_bio: link users when user decides to follow based on reading bio after reposting
        if user_link_strategy not in ["on_repost", "on_repost_bio", "on_repost_posts"]:
            raise Exception("Invalid user link strategy")
        self.user_link_strategy = user_link_strategy

        # Timeline selection strategy: how to select posts for the timeline
        # random: randomly select posts
        # random_weighted: randomly select posts with weights based on reposts and followers
        if timeline_select_strategy not in [
            "random",
            "random_weighted",
            "random_weighted_reversed",
            "bridging_attributes",
            "chronological",
            "other_partisan",
        ]:
            raise Exception("Invalid timeline selection strategy")
        self.timeline_select_strategy = timeline_select_strategy

        self.show_info = show_info

    def set_client(self, client: OpenAI | None):
        """
        Set the client for the platform.
        Default will be OpenAI's GPT-4o-mini for this project.
        """
        for user in self.users:
            user.llm = client

    def sample_user(self) -> Agent:
        """
        Returns a random user from the platform.
        Used for deciding which user is allowed to take an action in a simulation step.
        """
        return random.choice(self.users)

    def add_snapshot(self):
        """
        Create a snapshot of the network after each simulation step for analysis.
        """
        self.network_snapshots.append(
            {
                "users": [user.json(include_persona=False) for user in self.users],
                "connections": [link for link in self.user_links],
                "posts_reposts": {
                    post.post_id: post.reposts for post in self.raw_posts
                },
            }
        )

    def generate_posts_json(self):
        """
        Generate JSON object for all posts on the platform.
        """

        final_json = []

        for post in self.posts:
            final_json.append(
                {
                    "post_id": post["post_id"],
                    "user_id": post["user_id"],
                    "time": post["time"],
                    "post_content": post["post_content"].json(),
                }
            )

        return final_json

    def generate_users_json(self):
        """
        Generate JSON object for all users on the platform.
        """

        return [user.json(include_persona=True) for user in self.users]

    def generate_log(self):
        """
        Generate a log (JSON) of the platform for analysis.
        """

        total_input_tokens = sum([user.used_tokens_input for user in self.users])
        total_output_tokens = sum([user.used_tokens_output for user in self.users])
        total_cached_tokens = sum([user.used_tokens_cached for user in self.users])

        predicted_cost = ((0.6 / 1000000) * total_output_tokens) + (
            (0.15 / 1000000) * (total_input_tokens - total_cached_tokens)
            + ((0.075 / 1000000) * total_cached_tokens)
        )

        return {
            "total_tokens_input": total_input_tokens,
            "total_tokens_output": total_output_tokens,
            "total_tokens_cached": total_cached_tokens,
            "predicted_cost": predicted_cost,
            "users": self.generate_users_json(),
            "posts": self.generate_posts_json(),
            "raw_posts": [post.json() for post in self.raw_posts],
            "user_links": self.user_links,
            "actions": self.actions,
            "network_snapshots": self.network_snapshots,
        }

    def register_user(self, agent: Agent):
        """
        Add a user to the platform.
        """

        agent.identifier = len(self.users) + 1
        self.users.append(agent)

        agent.llm = get_client()

    def get_user(self, user_id: int) -> Agent:
        """
        Returns the user with the given user_id.
        Returns None if the user is not found.
        """
        for user in self.users:
            if user.identifier == user_id:
                return user
        return None

    def get_post(self, post_id: int) -> Post:
        """
        Returns the post with the given post_id.
        Returns None if the post is not found.
        """
        for post in self.posts:
            if post["post_id"] == post_id:
                return post["post_content"]

        return None

    def link_users(self, user_link_from: Agent, user_link_to: Agent):
        """
        Link two users on the platform.
        """

        # Link already exists
        if self.has_link(user_link_from.identifier, user_link_to.identifier):
            return

        # Don't allow self links
        if user_link_from == user_link_to:
            return

        # Register link and increase follower count
        self.user_links.append((user_link_from.identifier, user_link_to.identifier))
        user_link_to.increase_followers()

    def has_link(self, user_id_1: int, user_id_2: int) -> bool:
        """
        Returns True if the users are linked on the platform (user_id_1 follows user_id_2), else False.
        """
        return (user_id_1, user_id_2) in self.user_links

    def get_follower_count(self, user_id: int) -> int:
        """
        Get the number of followers of the user.
        """

        user = self.get_user(user_id)
        return user.followers

    def get_posts_of_user(self, user_id: int) -> list[Post]:
        """
        Return a list of all posts by the user.
        """

        return [post for post in self.posts if post["user_id"] == user_id]

    def pick_posts(self, posts, weights, size):
        """
        Pick posts based on weights.
        """

        picked_posts = []
        posts_left = posts

        for _ in range(size):
            if len(posts_left) == 0:
                break

            post = random.choices(posts_left, weights=weights)[0]
            picked_posts.append(post)

            # Remove posts if already picked
            for i in range(len(posts_left) - 1, -1, -1):
                if (
                    posts_left[i]["post_content"].post_id
                    == post["post_content"].post_id
                ):
                    del weights[i]

            posts_left = [
                p
                for p in posts_left
                if p["post_content"].post_id != post["post_content"].post_id
            ]

        return picked_posts

    def get_timeline_recommended_part(self, user_id: int, size: int = 5) -> list[dict]:
        """
        Create a part of the timeline with recommended posts.
        Strategies (self.timeline_select_strategy):
        - random: randomly select posts
        - random_weighted: randomly select posts with weights based on reposts and followers
        """

        # Get the id's of the users linked to the user
        linked_users = [link[1] for link in self.user_links if link[0] == user_id]

        # Get all posts of linked users
        posts_following = [
            post["post_content"].post_id
            for post in self.posts
            if post["user_id"] in linked_users
            and not post["post_content"].reposted_by(user_id)
            and not post["post_content"].author.identifier == user_id
        ]

        random_part = [
            post
            for post in self.posts
            if post["post_content"].post_id not in posts_following
            and not post["post_content"].author.identifier == user_id
            and not post["post_content"].reposted_by(user_id)
        ]

        # TODO: use raw_posts or posts????
        if self.timeline_select_strategy == "random":
            random_part = self.pick_posts(
                random_part, [1 for _ in random_part], min(size, len(random_part))
            )
            # random_part = random.sample(random_part, min(size, len(random_part)))
            return random_part
        elif self.timeline_select_strategy == "random_weighted":
            # Recent 50 posts
            random_part = random_part[-50:]

            if len(random_part) == 0:
                return []

            # Only keep distinct posts based on post['post_content'].post_id
            seen = set()
            result = []
            for post in random_part:
                if post["post_content"].post_id not in seen:
                    seen.add(post["post_content"].post_id)
                    result.append(post)
            random_part = result

            # Posts with more reposts are more likely to be recommended
            total_score = sum(
                [post["post_content"].reposts + 1 for post in random_part]
            )
            if total_score == 0:
                weights = [1 for _ in random_part]
            else:
                weights = [post["post_content"].reposts + 1 for post in random_part]

            random_part = self.pick_posts(
                random_part, weights, min(size, len(random_part))
            )
            print("Reposts: ", [post["post_content"].reposts for post in random_part])
            # random_part = random.choices(random_part, weights=weights, k=min(size, len(random_part)))

            return random_part
        elif self.timeline_select_strategy == "random_weighted_reversed":
            # Recent 50 posts
            random_part = random_part[-50:]

            if len(random_part) == 0:
                return []

            # Only keep distinct posts based on post['post_content'].post_id
            seen = set()
            result = []
            for post in random_part:
                if post["post_content"].post_id not in seen:
                    seen.add(post["post_content"].post_id)
                    result.append(post)
            random_part = result

            # Posts with less reposts are more likely to be recommended
            total_score = sum([post["post_content"].reposts for post in random_part])
            if total_score == 0:
                weights = [1 for _ in random_part]
            else:
                weights = [
                    (total_score + 1) - post["post_content"].reposts
                    for post in random_part
                ]

            random_part = self.pick_posts(
                random_part, weights, min(size, len(random_part))
            )
            return random_part

        elif self.timeline_select_strategy == "bridging_attributes":
            # Recent 50 posts
            # random_part.sort(key=lambda x: x["time"], reverse=True)
            random_part = random_part[-50:]

            if len(random_part) == 0:
                return []

            # Only keep distinct posts based on post['post_content'].post_id
            seen = set()
            result = []
            for post in random_part:
                if post["post_content"].post_id not in seen:
                    seen.add(post["post_content"].post_id)
                    result.append(post)
            random_part = result

            random_part.sort(
                key=lambda post: post["post_content"].bridging_score, reverse=True
            )
            return random_part[:5]
        elif self.timeline_select_strategy == "chronological":
            if len(random_part) == 0:
                return []

            # Only keep distinct posts based on post['post_content'].post_id
            seen = set()
            result = []
            for post in random_part:
                if post["post_content"].post_id not in seen:
                    seen.add(post["post_content"].post_id)
                    result.append(post)
            random_part = result

            random_part.sort(key=lambda x: x["time"], reverse=True)

            return random_part[:5]

        elif self.timeline_select_strategy == "other_partisan":
            # Recent 50 posts
            random_part = random_part[-50:]

            if len(random_part) == 0:
                return []

            # Only keep distinct posts based on post['post_content'].post_id
            seen = set()
            result = []
            for post in random_part:
                if post["post_content"].post_id not in seen:
                    seen.add(post["post_content"].post_id)
                    result.append(post)
            random_part = result

            k = 3

            # Posts with more reposts are more likely to be recommended
            # Boos for posts with more partisan difference
            total_score = sum(
                [post["post_content"].reposts + 1 for post in random_part]
            )
            if total_score == 0:
                weights = [1 for _ in random_part]
            else:
                weights = [
                    (post["post_content"].reposts + 1)
                    * np.log(
                        1
                        + k
                        + abs(
                            post["post_content"].author.persona["partisan"]
                            - self.get_user(user_id).persona["partisan"]
                        )
                    )
                    for post in random_part
                ]

            random_part = self.pick_posts(
                random_part, weights, min(size, len(random_part))
            )

            return random_part

    def get_timeline(self, user_id: int, size: int) -> list[dict]:
        """
        Creates a timeline for a user. It consists of:
        - 5 most recent posts by users than the user follows
        - 5 posts from the platform recommended by the platform (with a strategy set in self.timeline_select_strategy)
        """

        # Get the id's of the users linked to the user
        linked_users = [link[1] for link in self.user_links if link[0] == user_id]

        # Only show posts and reposts by linked users
        # Exclude posts that are already reposted by the user
        following_part = [
            post
            for post in self.posts
            if post["user_id"] in linked_users
            and not post["post_content"].reposted_by(user_id)
            and not post["post_content"].author.identifier == user_id
        ]

        # Sort timelime by time
        # following_part.sort(key=lambda x: x["time"], reverse=True)

        # Combine 5 posts of linked users with 5 recommended posts
        timeline = following_part[-5:] + self.get_timeline_recommended_part(
            user_id, size=5
        )
        timeline.sort(key=lambda x: x["time"], reverse=True)

        return timeline

    def post(self, user: Agent, content: str):
        """
        User posts a message.
        """

        timestamp = datetime.now()
        post = Post(
            len(self.posts) + 1,
            user,
            timestamp,
            content,
            show_info=self.show_info,
            calculate_bridging=self.timeline_select_strategy == "bridging_attributes",
        )

        self.raw_posts.append(post)

        self.posts.append(
            {
                "post_id": post.post_id,
                "user_id": user.identifier,
                "time": timestamp,
                "post_content": post,
            }
        )

    def repost(self, user: Agent, post_id: int):
        """
        User reposts a message.
        """

        timestamp = datetime.now()
        post = self.get_post(post_id)

        if post.author.identifier == user.identifier:
            raise Exception(
                f"User {user.identifier} tries to repost its own post {post_id}!"
            )

        if post.reposted_by(user.identifier):
            raise Exception(
                f"User {user.identifier} has already reposted post {post_id}!"
            )

        post.count_repost(user.identifier)

        if self.user_link_strategy == "on_repost":
            self.link_users(user, post.author)
        elif self.user_link_strategy == "on_repost_bio":
            print("Getting user's last posts")
            user_last_posts = self.get_posts_of_user(post.author.identifier)
            print("Asking user to link based on bio")
            should_link, explanation = user.link_with_user(
                post.author,
                post.content,
                user_last_posts,
                use_bio=True,
                use_follower_count=self.show_info,
            )
            if should_link:
                self.link_users(user, post.author)
                print(f"User {user.identifier} linked to user {post.author.identifier}")
                print(f"Explanation: {explanation}")
            else:
                print(
                    f"User {user.identifier} chose not to link to user {post.author.identifier}"
                )
                print(f"Explanation: {explanation}")
        elif self.user_link_strategy == "on_repost_posts":
            user_last_posts = self.get_posts_of_user(post.author.identifier)
            should_link, explanation = user.link_with_user(
                post.author, post.content, user_last_posts, use_bio=False
            )
            if should_link:
                self.link_users(user, post.author)
                print(f"User {user.identifier} linked to user {post.author.identifier}")
                print(f"Explanation: {explanation}")
            else:
                print(
                    f"User {user.identifier} chose not to link to user {post.author.identifier}"
                )
                print(f"Explanation: {explanation}")

        self.posts.append(
            {
                "post_id": len(self.posts) + 1,
                "user_id": user.identifier,
                "time": timestamp,
                "post_content": post,
            }
        )

    def add_action(self, user_id: int, action: Action, success: bool, prompt: str):
        """
        Adds action to the platform for logging purposes.
        """
        self.actions.append(
            {
                "user_id": user_id,
                "action": action.option,
                "content": action.content,
                "success": success,
                # 'explanation': action.explanation,
                "prompt": prompt,
            }
        )

    def parse_and_do_action(self, user_id: int, action: Action, prompt: str) -> None:
        """
        Based on the action chosen by the user, perform the action.
        """

        agent = self.get_user(user_id)

        if not agent:
            print("User not found")
            self.add_action(user_id, action, False, prompt)
            return

        if action.option == 2:
            self.post(agent, action.content)
        elif action.option == 1:
            try:
                self.repost(agent, int(action.content))
            except Exception as e:
                print("Invalid post ID: ", e)
                self.add_action(user_id, action, False, prompt)
                return
        elif action.option == 3:
            pass
        else:
            print("Invalid action")
            self.add_action(user_id, action, False, prompt)

        self.add_action(user_id, action, True, prompt)
