import streamlit as st

st.title("Finansal Danışman Chatbot")

user_query = st.text_input("Sorunuzu yazınız:")

if st.button("Gönder"):
    if user_query.strip():
        # Modelden cevap al
        response = rag_chain.invoke({"input": user_query})
        st.write("**Cevap:**")
        st.write(response["answer"])
    else:
        st.warning("Lütfen bir soru yazınız.")