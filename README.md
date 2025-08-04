# eNRRCrew: Accelerating eNRR catalyst Design through Multi-Agent Collaboration and Automated Structure-Activity Analysis

![Graphical Abstract](https://github.com/nkuhuxu/eNRRCrew/blob/main/images/TOC.png)

The electrocatalytic nitrogen reduction reaction (eNRR) represents a promising approach for sustainable ammonia production. However, understanding structure-activity relationships remains challenging due to the vast literature and complex data analysis required. Here, we present eNRRCrew, a novel multi-agent collaborative framework that integrates large language models (LLMs), machine learning techniques, and automated data analysis tools to advance eNRR research. The eNRRCrew comprises five agents, an orchestrator, a yield predictor, a Faradaic efficiency predictor, a GraphRAG retriever, and a CSV file handler. Users interact with eNRRCrew through the user interface provided by the Streamlit library to perform retrieval and prediction of structure-activity relationships in eNRR. 

 - **eNRR Yield predictor and FE predictor:** - Using pre-trained machine learning models in the former section to predict eNRR yield and FE.
 - **GraphRAG retriever:** - Enhancing responses by retrieving information from curated databases containing eNRR abstracts.
 - **CSV file handler:** - Writes and executes code to interact with CSV files obtained from text-mining workflow in response to user queries.

![Main Interface](https://github.com/nkuhuxu/eNRRCrew/blob/main/images/Main_Interfacce.png)

## Online Demo

Try eNRRCrew on [Online demo](https://enrrcrew.streamlit.app/). Users can interact with eNRRCrew through an intuitive user interface provided by the Streamlit library, enabling them to efficiently retrieve and predict structure-activity relationships in eNRR. 


## Useful Links 🔗

- eNRRCrew [demo video](https://youtu.be/KP-TBl0QJcY)
- Microsoft's GraphRAG [GraphRAG](https://github.com/microsoft/graphrag)
- Microsoft's AutoGen [AutoGen](https://github.com/microsoft/autogen)
- Streamlit [Streamlit](https://streamlit.io/)
- Microsoft's GraphRAG + AutoGen + Ollama + Chainlit = Fully Local & Free Multi-Agent RAG Superbot [Medium.com](https://medium.com/@karthik.codex/microsofts-graphrag-autogen-ollama-chainlit-fully-local-free-multi-agent-rag-superbot-61ad3759f06f) 📚



## 📦 Installation and Setup 

Follow these steps to set up and run eNRRCrew:

1. **Create conda environment and install python packages:**
    ```bash
   conda create -n eNRRCrew python=3.12.7
   conda activate eNRRCrew
   git clone https://github.com/nkuhuxu/eNRRCrew.git
   cd eNRRCrew
   pip install -r requirements.txt
    ```    

2. **Run eNRRCrew:**
    ```bash
    streamlit run appUI.py
    ```                


