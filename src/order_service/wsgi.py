"""_summary_
This module initializes and runs the Order Service application.
It imports the `create_app` function from the `order_service.app` module to create 
an instance of the application. The application instance is then run with debugging 
enabled if this module is executed as the main program.

Author:
    @TheBarzani
"""
from order_service.app import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting Order Service...")
    app.run(debug=True)
    print("Order Service started")
