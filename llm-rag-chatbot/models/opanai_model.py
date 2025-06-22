from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
# API anahtarını oku
api_key = os.getenv("OPENAI_API_KEY")
print(api_key[:10], "***")
# Excel dosyasını oku
df = pd.read_excel("finansal_danisman_veriseti_1000.xlsx")
# Satırları belgeye dönüştür
from langchain_core.documents import Document

docs = []
for idx, row in df.iterrows():
    content = f"Intent: {row['intent']}\nKullanıcı: {row['kullanici_mesaji']}\nBot: {row['bot_cevabi']}"
    doc = Document(page_content=content, metadata={"satir": idx})
    docs.append(doc)

print("Toplam belge sayısı:", len(docs))
# OpenAI embedding kullan
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # veya text-embedding-3-large
vector = embeddings.embed_query("hello, world!")
print(vector[:5])
from langchain_community.vectorstores import Chroma

# Vectorstore oluştur
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# Test sorgusu
retrieved_docs = retriever.invoke("Vergi iadesi nasıl alınır?")
print(len(retrieved_docs))
print(retrieved_docs[3].page_content)
# LLM modelini OpenAI ile tanımla
llm = ChatOpenAI(
    model="gpt-3.5-turbo",   # Daha güçlü istersen "gpt-4" yaz
    temperature=0.3,
    max_tokens=500
)
# Sistem mesajı tanımı
system_prompt = (
    "Sen bir finansal danışmansın ve kullanıcılardan gelen soruları yanıtlamakla görevlisin. "
    "Aşağıda verilen içerikleri kullanarak en doğru cevabı oluştur. "
    "Eğer bilgi yetersizse, 'Bu konuda net bir bilgim yok' şeklinde dürüstçe belirt. "
    "Yanıtlarını kısa, açık ve en fazla üç cümle olacak şekilde tut. "
    "Samimi, yönlendirici ve güven veren bir dil kullan.\n\n"
    "{context}"
)
# Prompt oluştur
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
# Zincirleri oluştur
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# Sorgu yap
response = rag_chain.invoke({"input": "Vergi iadesi nasıl alınır?"})
print(response)
print(response["answer"])
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Veri oku
df = pd.read_excel("finansal_danisman_veriseti_1000.xlsx")

# Veri split (örnek %10 test)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

y_true = []
y_pred = []

def extract_intent_from_answer(answer, intents):
    answer_lower = answer.lower()
    for intent in intents:
        if intent.lower() in answer_lower:
            return intent
    return "unknown"

# Model, retriever, rag_chain önceden oluşturulmuş ve hazır olmalı

intents = df['intent'].unique()

for _, row in test_df.iterrows():
    input_text = row['kullanici_mesaji']
    true_intent = row['intent']
    
    response = rag_chain.invoke({"input": input_text})
    answer = response.get("answer", "")
    
    predicted_intent = extract_intent_from_answer(answer, intents)
    
    y_true.append(true_intent)
    y_pred.append(predicted_intent)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))