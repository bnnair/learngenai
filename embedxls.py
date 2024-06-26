from langchain_community.llms import ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.embeddings import OllamaEmbeddings

import os
import time
import pandas as pd

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


filepath = os.path.join("data", "idbi-2014.xlsx")
outpath = os.path.join("data", "idbi-2014.csv")

print("filepath--", filepath)

data = pd.read_excel(filepath, engine="openpyxl")
# data.to_csv(outpath, index=False)

start = time.time()
loader = UnstructuredExcelLoader(
    filepath)

print("loaded excel file in loader")
documents = loader.load()

print(documents)
end = time.time()

print("loading data time elapsed-----", end-start)

# # documents = chromaUtils.filter_complex_metadata(documents)
print("------------------------------------------------------")
start = time.time()

# Create embeddings
embeddings = OllamaEmbeddings(model="llama2")
query_result = embeddings.embed_documents(data)

print("embedding ::::::, ", query_result[5], "---- completed")
end = time.time()

print("embedding time elapsed-----", end-start)

start = time.time()
# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=embeddings,
)
end = time.time()

print("vectorstore-----", end-start)
question = "this is a home loan statement. I would like to know how much total amount i paid during the year? calculate only the amount against remark as Loan Recovery."
# chroma_db = Chroma.from_documents(
#     documents, embeddings, persist_directory="./chroma_db"
# )
print("vectorstore assigned")

print("-------------------------------------------------------------------------")
retriever = vectorstore.as_retriever()

local_model = ollama.Ollama(model="llama2")

prompt_template = PromptTemplate(
    input_variables=["context"],
    template="Given this context: {context}, please directly answer the question: {question}.",
)
print("prompt template------", prompt_template)


# Set up the question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    local_model,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
)
print("======================================================")
result = qa_chain({"query": question})

print("result -----:", result)


# print("embedding the excel sheet and putting it in a collection and storing it in chromadb")
# client = chromadb.Client()
# collection = client.create_collection(name="docs")


# # store each document in a vector embedding database
# for i, d in enumerate(documents):
#     print("number: ", i)
#     # print(d)
#     response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
#     embedding = response["embedding"]
#     # print(embedding)
#     collection.add(
#         ids=[str(i)],
#         embeddings=[embedding],
#         documents=[d]
#     )

# # RETRIEVE ----
# print("Retrieving from the vector database")
# # an example prompt
# prompt = "It is a home loan statement. what is the total amount paid as loan recovery?"
# # prompt = "What animals are llamas related to?"
# # generate an embedding for the prompt and retrieve the most relevant doc
# response = ollama.embeddings(
#     prompt=prompt,
#     model="mxbai-embed-large"
# )
# results = collection.query(
#     query_embeddings=[response["embedding"]],
#     n_results=1
# )

# data = results['documents'][0][0]
# print(data)

# # # GENERATE ---
# print("Generating the response for the prompt")
# # generate a response combining the prompt and data we retrieved in step 2
# output = ollama.generate(
#     model="llama2",
#     prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
# )

# print(output['response'])
