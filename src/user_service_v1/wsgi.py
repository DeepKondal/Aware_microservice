from user_service_v1.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)