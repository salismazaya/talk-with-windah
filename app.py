import streamlit as st, os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from datetime import datetime

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

VECTOR = None

def get_vectorstore():
    global VECTOR
    if not VECTOR:
        text_chunks = get_text_chunks(open('text.txt').read())
        embeddings = OpenAIEmbeddings()
        VECTOR = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    return VECTOR

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(max_tokens = 200)

    memory = ConversationSummaryMemory(
        ai_prefix = 'Windah AI',
        llm = llm,
        memory_key='chat_history', return_messages=True)
    vectorstore = get_vectorstore()

    chain = load_qa_chain(llm, verbose = True)

    def ask(question):
        docs = vectorstore.similarity_search(question)
        with get_openai_callback() as cb:
            keren = chain.run(input_documents = docs, question = f"""
    System Prompt:
    - Sekarang tanggal {datetime.now().strftime("%d-%m-%Y")}
    - Bertindak sebagai Windah Basudara. ganti kata tentang Brando Franco Windah menjadi saya
    - Kamu tidak bisa berinteraksi secara fisik
    - Jika seseorang menanyakan tentang orang kedua itu berarti sedang membahas Windah Basudara, jika seseorang membahas tentang orang pertama itu berarti sedang membahas dirinya sendiri
    - {memory.buffer}

    Question: {question}
    Answer:
    """)
            memory.save_context({'input':question}, {'output': keren})
            # bot.send_message('6145009249', question)

        print(cb)    
        return keren
    
    return ask


def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)
    if not getattr(st.session_state, 'chat_history'):
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(response)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Talk With Windah",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Tanya Tentang Windah Basudara")
    user_question = st.text_input("Pertanyaan kamu:")
    if user_question:
        if not getattr(st.session_state, 'conversation'):
            st.session_state.conversation = get_conversation_chain(get_vectorstore())
        
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
