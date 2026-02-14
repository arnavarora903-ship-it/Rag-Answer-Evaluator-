# import streamlit as st
# from langchain_openai import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.evaluation.qa import QAEvalChain

# def generate_response(
#     uploaded_file,
#     openai_api_key,
#     query_text,
#     response_text
# ):
#     #format uploaded file
#     documents = [uploaded_file.read().decode()]
    
#     #break it in small chunks
#     text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=0
#     )
#     texts = text_splitter.create_documents(documents)
#     embeddings = OpenAIEmbeddings(
#         openai_api_key=openai_api_key
#     )
    
#     # create a vectorstore and store there the texts
#     db = FAISS.from_documents(texts, embeddings)
    
#     # create a retriever interface
#     retriever = db.as_retriever()
    
#     # create a real QA dictionary
#     real_qa = [
#         {
#             "question": query_text,
#             "answer": response_text
#         }
#     ]
    
#     # regular QA chain
#     qachain = RetrievalQA.from_chain_type(
#         llm=OpenAI(openai_api_key=openai_api_key),
#         chain_type="stuff",
#         retriever=retriever,
#         input_key="question"
#     )
    
#     # predictions
#     predictions = qachain.apply(real_qa)
    
#     # create an eval chain
#     eval_chain = QAEvalChain.from_llm(
#         llm=OpenAI(openai_api_key=openai_api_key)
#     )
#     # have it grade itself
#     graded_outputs = eval_chain.evaluate(
#         real_qa,
#         predictions,
#         question_key="question",
#         prediction_key="result",
#         answer_key="answer"
#     )
    
#     response = {
#         "predictions": predictions,
#         "graded_outputs": graded_outputs
#     }
    
#     return response

# st.set_page_config(
#     page_title="Evaluate a RAG App"
# )
# st.title("Evaluate a RAG App")

# with st.expander("Evaluate the quality of a RAG APP"):
#     st.write("""
#         To evaluate the quality of a RAG app, we will
#         ask it questions for which we already know the
#         real answers.
        
#         That way we can see if the app is producing
#         the right answers or if it is hallucinating.
#     """)

# uploaded_file = st.file_uploader(
#     "Upload a .txt document",
#     type="txt"
# )

# query_text = st.text_input(
#     "Enter a question you have already fact-checked:",
#     placeholder="Write your question here",
#     disabled=not uploaded_file
# )

# response_text = st.text_input(
#     "Enter the real answer to the question:",
#     placeholder="Write the confirmed answer here",
#     disabled=not uploaded_file
# )

# result = []
# with st.form(
#     "myform",
#     clear_on_submit=True
# ):
#     openai_api_key = st.text_input(
#         "OpenAI API Key:",
#         type="password",
#         disabled=not (uploaded_file and query_text)
#     )
#     submitted = st.form_submit_button(
#         "Submit",
#         disabled=not (uploaded_file and query_text)
#     )
#     if submitted and openai_api_key.startswith("sk-"):
#         with st.spinner(
#             "Wait, please. I am working on it..."
#             ):
#             response = generate_response(
#                 uploaded_file,
#                 openai_api_key,
#                 query_text,
#                 response_text
#             )
#             result.append(response)
#             del openai_api_key
            
# if len(result):
#     st.write("Question")
#     st.info(response["predictions"][0]["question"])
#     st.write("Real answer")
#     st.info(response["predictions"][0]["answer"])
#     st.write("Answer provided by the AI App")
#     st.info(response["predictions"][0]["result"])
#     st.write("Therefore, the AI App answer was")
#     st.info(response["graded_outputs"][0]["results"])

# own CODE 
import streamlit as st 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.document_loaders import PyPDFLoader  # capital L in Loader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
import tempfile 
import os 
import math 

st.set_page_config(page_title= 'Rag Application Response Evaluator')
st.title('Is the RESPONSE Valid?')

api_key = st.sidebar.text_input(placeholder='sk-',label='OpenAI API Key',type='password')

uploaded_file = st.file_uploader(
    "Upload your document here",
    type = ['.txt','.pdf']
)

# & this character cannot be used in function name, it is not allowed. 
def main(uploaded_file,api_key):
    string_data = file_type_checker(uploaded_file)
    if string_data is None: # in python Booleans are capital True, False, None capital T, F , N :)
        st.error('Document loading error,restart the application')
        return None, None, None 
    else:
        vector_store,embeddings = text_splitter_and_VectorStore(string_data,api_key)
        llm = initialize_llm(api_key)
        retriever = vector_store.as_retriever()
        return llm, retriever,embeddings # here we have used lllm and retreiver outside this function with chains thus not returing them will give a 
    # Scope Error. Thus, we are returing these 2 variables 
    # so that they can be used outside the function as well. 


def file_type_checker(uploaded_file): # here if uploaded file is none or any unsupported file then 
# string data is never created, this will fail the entire code. Thus, to prevent this 
    string_data = None 
    if uploaded_file is not None:
        file_type = uploaded_file.type # this returns MIME type 'txt/plain' for .txt files and 'application/pdf' for .pdf files 
        filename = uploaded_file.name # this returns file name 
        file_extension = os.path.splitext(filename)[1].lower() # this splits the file name and from there we access element 1 of the splitted text list which is the file extension and further lower case it 
        if file_type == 'txt/plain' or file_extension == '.txt':
            string_data = uploaded_file.read().decode('UTF-8')
        elif file_type == 'application/pdf' or file_extension == '.pdf':
            with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name # this line gives path to our temporary file which exists in temporary directory 
            loader = PyPDFLoader(tmp_file_path)
            loaded_file = loader.load()
            string_data = '\n'.join(doc.page_content for doc in loaded_file) 
            os.unlink(tmp_file_path) # this line deleted the temporary file 
        else: 
            st.error('Unsupported File Type')
    return string_data 

def initialize_llm(api_key):
    llm = ChatOpenAI (openai_api_key = api_key,temperature=0.1,model='gpt-4.1-nano')
    return llm 
def initialize_judge_llm(api_key):
    judge_llm = ChatOpenAI (openai_api_key = api_key,temperature=0,model='gpt-4.1-nano')
    return judge_llm

def text_splitter_and_VectorStore(string_data,api_key):
    text_splitter = RecursiveCharacterTextSplitter (
        separators = ['\n','\n\n'], # RecursiverCharacterTextSplitter uses separators (plural)
        chunk_size = 400, # chunks_size is incorrect, correct argument is chunk_size (chunk is singular)
        chunk_overlap=75,
    )
    splitted_text = text_splitter.create_documents([string_data])
   
    embeddings = OpenAIEmbeddings(openai_api_key =api_key) # as here we are not using .env file, therefore passing api key directly here is recommend. 
    vector_store = FAISS.from_documents(splitted_text,embeddings)
    return vector_store, embeddings 


user_question = st.text_input('Ask your question ?')
final_answer = st.text_input('Please provide answer of the question.')

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '''You are a very intelligent and diligent assistant. 
        You answer the question asked based on retrieved context. If you do not know 
        the answer, you honestly say " I apologize, I do not know the answer of your question." You 
        do not hallucinate or invent answers that provided information does not support. 
        {context}'''),
        ('human','{input}')
    ]
)

judge_prompt = ChatPromptTemplate.from_messages([
    ('system','''You are playing role of a judge here. 2 different answers will be provided to you.Your duty 
    here is to see whether they are semantically correct, are they stating same facts, if dates are mentioned in them are they exactly same,
    Do not pass incorrect, if you find that different words are used or answers are paraphrased.
    Decide if the core idea coveyed by them is same.
    Example: I like apple , I am fond of apple. 
    In this example the words chosen are different. However, core meaning is same. 
    If both answers mean exactly same then you pass "Correct" as your judgement, if the meaning is even slightly different or you are not sure then you pass "Incorrect" as your judgement'''),

    ('human',
    '''rag_answer = {answer}
    user_answer = {final_answer}''')
])


if uploaded_file and api_key and user_question: 
    llm, retriever,embeddings = main (uploaded_file, api_key)
    feeding_chain = create_stuff_documents_chain(llm,prompt)
    wrapper_chain = create_retrieval_chain (retriever,feeding_chain)

    response = wrapper_chain.invoke({'input':user_question}) # create_retrieval_chain automatically passes context so while 
    # invoking wrapper_chain only human input has to be passed. 

    st.markdown('**App Answer:**')
    # answer = st.write(response['answer']) # create_retriveal_chain returns a DICTIONARY not a message object. 
    # st.write() returns none so storing that in a variable will always fail. 
    answer = response['answer']
    st.write(answer)
    #  final_answer == answer this is highly likely to fail as here 
    # we are comparing strings 
    # and even an extra space will make it fail 

    #Verification code try 1:

    # if final_answer.strip() == answer.strip(): # .strip() removes the leading and trailing whitespaces 
    #     st.success('The answer provided by the RAG Application is CORRECT')
    # else:
    #     st.error('The answer provided by the RAG Application is INCORRECT')
    

    # now with current verification code it is trying to match the strings literally and if their 
    # is even a slight mismatch then it will give error even though meaning of both the answers could be the same 

    # To improve this we have 2 options 
    # 1) Create another prompt with explicit instructions on what to do and then pass it to
    # llm again. Here llm is the judge, it will judge if the user and rag app generated asnwers mean the same or not. 
    # 2) Is to convert the user answer and the rag app answer to embeddings then compare 
    # their cosine similarity and if they match or exceed the developer given threshold then it will be a success otherwise retry 

    # We are choosing option 2 as it is less likely to create bugs becuase it is Maths and will be easier to debug if any errors 
    
    # Verification code try 2: 
    # converting user answer and rag app generated answer into embeddings 
    if not final_answer.strip():
        st.error('Please provide the correct answer')
        st.stop()
    else:
        user_answer_embeddings = embeddings.embed_query(final_answer)
        rag_answer_embeddings = embeddings.embed_query(answer)

        assert len(user_answer_embeddings)  == len(rag_answer_embeddings) # cosine similarity requires similar length vectors, if length is not equal cosine similarity 
        # will not work and the data will corrupt.    assert  immediately stops exceuciton if length is not same. 
        # in vector direction represents meaning. So if 2 vectors have same direction then it means that english words/sentences have same meaning. 
        def cosine_similarity(vec1,vec2): 
            dot_product = sum(a*b for a,b in zip(vec1,vec2))
            norm_vec1 = math.sqrt(sum(a*a for a in vec1)) # this is giving magnitude of the vector |A| , remember vectors from physics 
            norm_vec2 = math.sqrt(sum(a*a for a in vec2))
            return dot_product/(norm_vec1 * norm_vec2)

        similarity_score = cosine_similarity(user_answer_embeddings,rag_answer_embeddings)
        threshold_lower_limit = 0.65
        threshold_upper_limit = 0.9
        # implemeting llm as judge 
        # step 1) setting up range for cosine similarity based on which answers will be passed to llm (judge)
        # if similarity_score == threshold_lower_limit: 
        #     judge_llm = initialize_judge_llm(api_key)
        #     chain = judge_prompt|judge_llm
        #     final_verdict = chain.invoke(answer,final_answer)
        #     st.write(final_verdict['answer'])
        # elif threshold_lower_limit < similairty_score < threshold_upper_limit:
        #     judge_llm = initialize_judge_llm(api_key)
        #     chain = judge_prompt|judge_llm
        #     final_verdict = chain.invoke(answer,final_answer)
        #     st.write(final_verdict['answer'])
        # elif similairty_score >= threhold_upper_limit:
        #     judge_llm = initialize_judge_llm(api_key)
        #     chain = judge_prompt|judge_llm
        #     final_verdict = chain.invoke(answer,final_answer)
        #     st.write(final_verdict['answer'])
           
        # Better way to write above conditonal code 
        
        def classify_similarity(similarity_score,threshold_lower_limit,threshold_upper_limit):
            if similarity_score < threshold_lower_limit: # float numbers are almost never equal so instead of using == it is better to use <= or >=
                return 'Low'
            elif similarity_score > threshold_lower_limit and similarity_score <= threshold_upper_limit:
                return 'Medium' 
            elif similarity_score >= threshold_upper_limit:
                return 'High' 
        zone = classify_similarity(similarity_score,threshold_lower_limit,threshold_upper_limit)
        if zone in ['Medium', 'High']:
            
            judge_llm = initialize_judge_llm(api_key)
            chain = judge_prompt|judge_llm
            final_verdict = chain.invoke({'answer':answer,'final_answer':final_answer})
            st.write(final_verdict.content)
            st.write(similarity_score)
        elif zone == 'low':
            st.write(similarity_score)
            st.write('Cosine similarity is below threshold')
            st.write('rag_answer:',answer) # here either use a comma to separate rag_answer and answer or use a f string. 



     
            