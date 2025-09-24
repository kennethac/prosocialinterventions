import dotenv
import os
import json
import pickle
import random

from openai import OpenAI

from Agent import Agent, get_client
from Platform import Platform
from NewsFeed import NewsFeed

dotenv.load_dotenv()

import kagglehub

# Download latest version
path = kagglehub.dataset_download("rmisra/news-category-dataset")

print("Path to dataset files:", path)


def log_action(user, action):
    """
    Log the action taken by the user to the console.
    """

    log_msg = f"User {user.identifier} chose action "

    if action.option == 1:
        log_msg += "1, repost."
        log_msg += f"User reposted message with ID {action.content}\n"
    elif action.option == 2:
        log_msg += "2, post.\n"
        log_msg += f"User wrote: {action.content}\n"
    elif action.option == 3:
        log_msg += "3, do nothing.\n"
    else:
        log_msg += f"{action.option}, which is invalid.\n"

    return log_msg


def select_users(persona_path, n):
    """
    Create a sample of users for the simulation from the persona file.
    """

    # According to Gallup, 45% of Americans identify as Democrats, 46% as Republicans, and 9% as other (2025)
    fraction_democrat = 0.45
    fraction_republican = 0.46
    fraction_non_partisan = 0.09

    users = json.load(open(persona_path, "r"))

    democrat_users = [user for user in users if user["party"] == "Democrat"]
    republican_users = [user for user in users if user["party"] == "Republican"]
    non_partisan_users = [user for user in users if user["party"] == "Non-partisan"]

    # Randomly sample users from each group
    democrat_sample = random.sample(democrat_users, int(n * fraction_democrat))
    republican_sample = random.sample(republican_users, int(n * fraction_republican))
    non_partisan_sample = random.sample(
        non_partisan_users, int(n * fraction_non_partisan)
    )

    return democrat_sample + republican_sample + non_partisan_sample


def run_simulation(
    simulation_size=500,
    simulation_steps=10000,
    user_link_strategy="on_repost_bio",
    timeline_select_strategy="random_weighted",
    show_info=True,
    run_nr=1,
):
    # Define the path to the persona file
    persona_path = os.path.join(os.getcwd(), "personas.json")
    news_feed = NewsFeed(path + "/News_Category_Dataset_v3.json")

    filename = f"../results/{user_link_strategy}_{timeline_select_strategy}_{'info' if show_info else 'noinfo'}_{run_nr}"

    platform = Platform(
        user_link_strategy=user_link_strategy,
        timeline_select_strategy=timeline_select_strategy,
        show_info=show_info,
    )

    # Ensure the right fraction of Democrats, Republicans, and non-partisans
    selected_users = select_users(persona_path, n=simulation_size)

    # Set client for platform to OpenAI gpt-4o-mini
    model = os.environ["SELECTED_MODEL"]
    client = get_client()

    # Register users
    [platform.register_user(Agent(model, user)) for user in selected_users]
    platform.set_client(client)

    try:
        for i in range(simulation_steps):
            print(f"Simulation step {i + 1}")

            # Select a random user
            user = platform.sample_user()

            # Perform an action
            action, prompt = user.perform_action(
                news_feed.get_random_news(10),
                platform.get_timeline(user.identifier, 10),
            )
            platform.parse_and_do_action(user.identifier, action, prompt)

            print(log_action(user, action))

            # Add snapshot of the platform for analysis
            platform.add_snapshot()

            # Refresh client every 1000 steps
            if i % 1000 == 0 and i != 0:
                new_client = get_client()
                platform.set_client(new_client)
                client.close()

                client = new_client
    except:
        json.dump(
            platform.generate_log(),
            open(filename + ".json", "w"),
            indent=4,
            default=str,
        )

        # Set reuse of platform
        platform.set_client(None)
        client.close()

        pickle.dump(platform, open(filename + ".pkl", "wb"))

    json.dump(
        platform.generate_log(), open(filename + ".json", "w"), indent=4, default=str
    )

    # Set reuse of platform
    platform.set_client(None)
    client.close()

    pickle.dump(platform, open(filename + ".pkl", "wb"))


if __name__ == "__main__":
    # Run five simulations
    for i in range(1, 6):
        print(f"Running simulation {i}...")

        run_simulation(
            simulation_size=500,
            simulation_steps=10000,
            user_link_strategy="on_repost_bio",
            timeline_select_strategy="other_partisan",
            show_info=True,
            run_nr=i,
        )
