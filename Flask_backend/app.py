from services.llm_service import generate_response
from flask import Flask
from flask_cors import CORS

def create_app():
    app=Flask(__name__)
    CORS(app,origins=["http://localhost:4200",
                "http://127.0.0.1:4200",
                "http://192.168.0.126:4200"])
    # history=[]
    # while True:
    #     user_input=input("you: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     history.append(f"""user: {user_input} AI: {generate_response(user_input,history)}""")
    #     print (history[-1].split("AI: ")[-1].strip())
    from routes.chat_routes import chat_route
    app.register_blueprint(chat_route,url_prefix="/api/ai")

    return app

if __name__=="__main__":
    app=create_app()
    app.run()