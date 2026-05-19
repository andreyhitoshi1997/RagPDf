import json

NOTEBOOK_PATH = "RagCode_RAG.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue

    cell_id = cell.get("id", "")

    # -------------------------------------------------------
    # Célula qa-chain: usa langchain_classic (correto para LangChain 1.x)
    # -------------------------------------------------------
    if cell_id == "qa-chain":
        new_source = [
            "# Monta a cadeia de QA — LangChain >= 1.x (usa langchain_classic)\n",
            "from langchain_classic.chains import create_retrieval_chain\n",
            "from langchain_classic.chains.combine_documents import create_stuff_documents_chain\n",
            "\n",
            "# Documento-chain (combina documentos usando o prompt)\n",
            "doc_chain = create_stuff_documents_chain(\n",
            "    llm=ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_key),\n",
            "    prompt=prompt\n",
            ")\n",
            "\n",
            "# Retrieval-chain (recupera documentos relevantes e os passa ao doc_chain)\n",
            "qa = create_retrieval_chain(\n",
            "    retriever=retriever,\n",
            "    combine_documents_chain=doc_chain\n",
            ")\n",
        ]
        cell["source"] = new_source
        cell["outputs"] = []
        cell["execution_count"] = None
        print("✅ Célula 'qa-chain' corrigida com langchain_classic.")

    # -------------------------------------------------------
    # Célula run-query: usa qa.invoke() em vez do depreciado qa.run()
    # -------------------------------------------------------
    elif cell_id == "run-query":
        new_source = [
            "# Exemplo de consulta ao código do repositório\n",
            "query = 'Qual a estrutura da classe ClinicListScreen?'\n",
            "result = qa.invoke({'input': query})\n",
            "print(result['answer'])\n",
        ]
        cell["source"] = new_source
        cell["outputs"] = []
        cell["execution_count"] = None
        print("✅ Célula 'run-query' corrigida (qa.run → qa.invoke).")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("\n✅ Notebook salvo com sucesso!")
