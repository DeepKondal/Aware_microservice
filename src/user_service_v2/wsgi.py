"""_summary_
This module initializes and runs the User Service application (v2).
It imports the `create_app` function from the `user_service_v2.app` module to create 
an instance of the application. The application instance is then run with debugging 
enabled if this module is executed as the main program.

Author:
    @TheBarzani
"""
from user_service_v2.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
