### Setup
1. Create the virtual env with 
    ```
    conda create -n <env_name>
    conda activate <env_name>
    pip install -r requirements.txt
    ```

2. Create a Serper account at https://serper.dev/ and generate an API key (free trial comes with 2500 credits)
3. Run `crawl4ai-setup` to install needed browser dependencies. If needed, check the [Crawl4AI Documentation](https://docs.crawl4ai.com/core/installation/).
4. Add a `.env` file to the `server` directory with needed API keys. e.g. 
    ```
    SERPER_API_KEY=...
    GOOGLE_API_KEY=...
    OPENAI_API_KEY=...
    NCBI_API_KEY=...
    ```

### Running app
1. Start the FastAPI server, which will run on localhost:8000
    ```
    cd server
    python main.py
    ```

2. Serve the frontend, which will run on localhost:5173
    ```
    cd client
    npm install
    npm run dev
    ```

### Other settings
- The model name, max token count, and max tool calls used per response are set at the top of `server/main.py`
- Prompts are located in `server/aso_workflow/`