from flask import Blueprint,request,jsonify
from services.llm_service import generate_response
from processors.image_processor import extract_text_from_image
from processors.pdf_processor import extract_text_from_pdf_file
from processors.docx_processor import extract_text_from_docx_file

chat_route=Blueprint('chat',__name__)

@chat_route.route('/send_chat',methods=['POST'])
def chat_method():
    text_message=request.form.get('user_chat')
    chat_history=request.form.get('chat_history')
    chat_attachments=request.files.getlist('attachments')
    selected_mode=request.form.get('selected_mode')
    extracted_input=""

    for attachments in chat_attachments:
        if files.content_type.startswith("image"):
            extracted_input +=extract_text_from_image(files)
        elif files.filename.endswith('.pdf'):
            extracted_input+=extract_text_from_pdf_file(files)
        elif files.filename.endswith(".docx"):
            extracted_input+=extract_text_from_docx_file(files)
        files.seek(0)


    return jsonify(generate_response(user_input=text_message,history=chat_history,mode=selected_mode,file=extracted_input)), 200

