import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

import ollama
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# Excel'den veriyi oku
df = pd.read_excel("finansal_danisman_veriseti_1000.xlsx")
# Her satırı Document objesine çevir
docs = [
    Document(
        page_content=f"Intent: {row['intent']}\nKullanıcı: {row['kullanici_mesaji']}\nBot: {row['bot_cevabi']}",
        metadata={"satir": idx}
    )
    for idx, row in df.iterrows()
]
print(f"Toplam belge sayısı: {len(docs)}")
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Modeli yükle
llm = ChatOllama(model="llama3.2", temperature=0.4)

# Prompt şablonunu oluştur
system_prompt = (
    "Sen bir finansal danışmansın ve kullanıcılardan gelen soruları yanıtlamakla görevlisin. "
    "Eğer konu finansla alakalı değilse, sadece 'Bu konuda yardımcı olamıyorum' şeklinde yanıt ver. "
    "Yanıtlarını kısa, açık ve en fazla üç cümle olacak şekilde tut."
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Zincirleri oluştur
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Sorgu çalıştır
response = rag_chain.invoke({"input": "Vergi iadesi nasıl alınır?"})
print(response.get("answer", "Beklenen formatta yanıt alınamadı!"))
import ipywidgets as widgets
from IPython.display import display

# Chat arayüzü oluştur
chat_output = widgets.Output()
input_text = widgets.Text(placeholder='Sorunuzu yazın...', layout=widgets.Layout(width='80%'))
send_button = widgets.Button(description='Gönder', button_style='success')

def on_button_clicked(b):
    with chat_output:
        user_input = input_text.value
        if user_input:
            print(f"👤: {user_input}")
            response = rag_chain.invoke({"input": user_input})
            answer = response.get("answer", "Yanıt alınamadı")
            print(f"🤖: {answer}\n")
            input_text.value = ''  # Input alanını temizle

send_button.on_click(on_button_clicked)

# Enter tuşu desteği
def handle_submit(sender):
    on_button_clicked(sender)

input_text.on_submit(handle_submit)

# Arayüzü göster
display(widgets.VBox([
    widgets.HTML("<h2>Finansal Danışman Chatbotu</h2>"),
    chat_output,
    widgets.HBox([input_text, send_button])
]))
# Veri oku
df = pd.read_excel("finansal_danisman_veriseti_1000.xlsx")

# Veri split 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
y_true = []
y_pred = []

def extract_intent_from_answer(answer, intents):
    answer_lower = answer.lower()
    for intent in intents:
        if intent.lower() in answer_lower:
            return intent
    return "unknown"
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