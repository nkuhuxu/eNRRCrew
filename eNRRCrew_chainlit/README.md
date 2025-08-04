# 🤖eNRRCrew: Accelerating eNRR Catalyst Design through Multi-Agent Collaboration and Automated Structure-Activity Analysis

**eNRRCrew** is a novel multi-agent collaborative framework designed to accelerate research in the electrocatalytic nitrogen reduction reaction (eNRR). By integrating Large Language Models (LLMs), machine learning, and automated data analysis, eNRRCrew facilitates the understanding of structure-activity relationships in eNRR, a promising approach for sustainable ammonia production.

This framework is composed of five specialized agents that work in concert: an orchestrator, a yield predictor, a Faradaic efficiency (FE) predictor, a GraphRAG retriever, and a CSV file handler. Users can interact with eNRRCrew through a user-friendly interface to perform retrieval and prediction tasks, streamlining the analysis of complex eNRR data.

 - **eNRR Yield predictor and FE predictor:** Utilizes pre-trained machine learning models to predict eNRR yield and FE based on input data.
 - **GraphRAG retriever:** Enhances responses by retrieving relevant information from a curated knowledge base of eNRR abstracts and providing novel electrocatalyst recommendations.
 - **CSV file handler:** Executes code to interact with CSV files, enabling seamless data manipulation and analysis in response to user queries.

The complete dataset of structure-activity relationships, extracted from 2,321 analyzed papers, can be found in `data_include_morphology_electrocatalyst.csv`.


## 🚀 Online Demo

Experience eNRRCrew live on our [Online Demo](https://enrrcrew.streamlit.app). The intuitive interface allows for efficient retrieval and prediction of structure-activity relationships in eNRR. 


## 🔗 Useful Links

- eNRRCrew [demo video](https://youtu.be/KP-TBl0QJcY)
- Microsoft's GraphRAG [GraphRAG](https://github.com/microsoft/graphrag)
- Microsoft's AutoGen [AutoGen](https://github.com/microsoft/autogen)
- Streamlit [Streamlit](https://streamlit.io/)
- Microsoft's GraphRAG + AutoGen + Ollama + Chainlit = Fully Local & Free Multi-Agent RAG Superbot [Medium.com](https://medium.com/@karthik.codex/microsofts-graphrag-autogen-ollama-chainlit-fully-local-free-multi-agent-rag-superbot-61ad3759f06f) 📚



## 📦 Installation and Setup 

Follow these steps to set up and run eNRRCrew on your local machine.

1. **Create a Conda Environment and Install Dependencies**
First, create a dedicated conda environment to manage the project's dependencies. Then, clone the repository and install the required Python packages.
    ```bash
    conda create -n eNRRCrew python=3.12.7
    conda activate eNRRCrew
    git clone https://github.com/nkuhuxu/eNRRCrew.git
    cd eNRRCrew
    cd eNRRCrew_chainlit
    pip install -r requirements.txt
    ```    

2. **Set Up API Keys**
The application requires API keys for OpenAI services. You will need to configure these in `appUI.py` and `.env`. Locate the following sections in `appUI.py` and replace the placeholder text with your actual API keys and base URLs:

    ```bash
    # Setup OpenAI API client
    client = OpenAI(
        api_key="your_api_key_here",
        base_url="your_api_base_url_here"
    )

    # ...

    llm_config_autogen = {
        "seed": 42,
        "temperature": 0,
        "config_list": [{"model": "gpt-4.1-mini",
                        "base_url": "your_api_base_url_here",
                        'api_key': 'your_api_key_here'}],
        "timeout": 60,
    }

    llm_manager = {
        "seed": 42,
        "temperature": 0,
        "config_list": [{"model": "gpt-4.1",
                        "base_url": "your_api_base_url_here",
                        'api_key': 'your_api_key_here'}],
        "timeout": 60,
    }
    ```
3. **Run eNRRCrew**
Once the dependencies are installed and the API keys are configured, you can run the application using Chainlit.
    ```bash
    chainlit run appUI.py
    ```    
## 📖 Usage
eNRRCrew offers a versatile interface for interacting with eNRR data. Here are the main functionalities:
1. **Querying with GraphRAG**
You can ask questions about eNRR, and the Retriever Agent will use GraphRAG to find relevant information from the knowledge graph.
 - **Example Query:** Which catalysts exhibit a high yield per unit mass in the electrocatalytic nitrogen reduction reaction?

2. **Interacting with the Dataset**
The CSV Handler Agent allows you to query the `data_include_morphology_electrocatalyst.csv` dataset. You can ask for specific data points, summaries, or insights.
 - **Example Query:** What is the average applied electrode potential for eNRR yield from the csv file?

3. **Predicting NH3 Yield and Faradaic Efficiency**
The **Yield Predictor** and **FE Predictor** agents can predict the performance of electrocatalyst systems. You can provide a description of the system, and the agents will process the text, create the necessary input files, and return the predictions.
 - **Example Query:** Please predict the yield and Faradaic efficiency based on the following description: An original ammonia electrocatalyst system for nitrogen reduction reaction (NRR) could be designed using a bimetallic single-atom catalyst composed of cobalt (Co) and molybdenum (Mo) atoms dispersed on a nitrogen-doped porous carbon nanofiber (PCNF) matrix. This morphology features high surface area porous nanofibers to maximize active site exposure and facilitate nitrogen adsorption and activation, leveraging the synergistic effect of Co and Mo to enhance NRR activity and selectivity while suppressing competing hydrogen evolution reaction (HER). The catalyst should operate at a mild applied potential around -0.3 V vs. reversible hydrogen electrode (RHE), balancing sufficient driving force for nitrogen reduction with minimized side reactions. The electrolyte environment may be mildly acidic (pH 5) to optimize proton availability and suppress HER, using a buffered aqueous solution such as 0.1 M phosphate buffer.

4. **Potential Catalyst Recommendation**
You can engage in dialogue with eNRRCrew to request recommendations for novel catalyst systems.
 - **Example Query:** Based on literature retrieval, propose an ammonia electrocatalyst system that has not been previously reported, specifying the catalyst's elemental composition, morphology, electrode potential under reaction conditions, pH, and other relevant parameters, ensuring that the description is original and not directly copied from existing sources.
