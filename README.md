# Updated project containing a more reproducible version of the code published by the authors of for paper "Can We Fix Social Media? Testing Prosocial Interventions using Generative Social Simulation"

This project contains a fork of the original code at [cssmodels/prosocialinterventions](https://github.com/cssmodels/prosocialinterventions) with a few improvements to make it easier to get started. For example:

1. Uses Ollama for quicker local testing.
2. Uses `uv` to manage dependencies and virtual environment
3. Uses `kagglehub` to retrieve the news data set automatically at runtime.

## How to run?

1. Fill in the necessary API keys in a `.env` file:

   ```plaintext
   OPENAI_API_KEY=YOUR_KEY
   PERSPECTIVE_API_KEY=YOUR_KEY
   ```

2. Edit the main script to set the size of the simulation, the number of steps and strategies in the call to the function `run_simulation` in `main.py`:

   ```python
   run_simulation(simulation_size=500, simulation_steps=10000, 
                       user_link_strategy="on_repost_bio", 
                       timeline_select_strategy="other_partisan",
                       show_info=True, run_nr=i)
   ```

3. Run the main script:

   ```bash
   uv run main.py
   ```

Outputs will be saved in results folder - a Pickle file with the whole platform state and a JSON file with results.
