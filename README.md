# How to use

1. Download the repo.
2. Make sure you have all necessary packages installed, create an LLM environemtn if needed, especially if you want to try out future projects like this one.
3. Everything should be set up after this, simply open a terminal/cmd prompt in the folder and type `streamlit run rag.py`
4. A window will automatically pop up with the chatbot.

To get started with development, follow these steps:

1. Ensure you have Python installed on your machine. This project requires Python 3.12.1 or later.

2. Clone the repository to your local machine using Git:

   ```
   git clone https://github.com/E3-Consulting/sql-agent-poc.git
   ```

3. Navigate into the cloned repository:

   ```
   cd sql-agent-poc
   ```

4. This project uses Poetry for dependency management. If you don't have Poetry installed, install it by following the instructions here: https://python-poetry.org/docs/#installation

5. Install the project dependencies by running:

   ```
   poetry install
   ```

6. Set up your environment variables. Copy the `.env.example` file to a new file named `.env` and fill in your API keys:

   ```
   cp .env.example .env
   # Now edit the .env file with your favorite text editor
   ```

7. To activate the virtual environment created by Poetry, run:

   ```
   poetry shell
   ```

8. You're now ready to start development. To run the application locally, use:

   ```
   poetry run streamlit run app.py
   ```

9. To contribute to the project, make sure to create a new branch for your feature or fix and submit a pull request once you're done.

Happy coding!

# gcloud setup

1. Follow installation instructions here- https://cloud.google.com/sdk/docs/install 

2. Then, follow the "Configure ADC with your Google Account" guide here- https://cloud.google.com/docs/authentication/provide-credentials-adc 

3. In .env file make sure to have this line: 

```
GOOGLE_APPLICATION_CREDENTIALS="path/to/file/directory/application_default_credentials.json"
```