# Testing Types
- Unit Testing
- Integration Testing
- Contract Testing
- End-to-End (E2E) Testing

## Unit Tests: Testing in Isolation
Unit tests focus on testing individual units of code in isolation which can be functions, methods, or classes, typically representing the smallest testable parts of an application. The key principle of unit testing is to isolate each unit from the rest of the codebase and verify its behavior independently.
- In the context of Aware-Microservices, this means testing 

## Component Tests: Testing Interactions
Component tests validate the interactions and integration between multiple units or components within a module or subsystem. External dependencies may be mocked or stubbed to isolate the component under test.

## Integration Tests: Testing the System as a Whole
Integration tests evaluate the interaction and integration of multiple modules or subsystems to ensure that they function correctly when combined. Unlike unit tests and component tests, which focus on isolated parts of the system, integration tests validate the behavior of the system as a whole.

> ### Test Doubles: Mocks and Stubs
> Test doubles are objects that stand in for real objects during testing. They are used to simulate the behavior of real objects in controlled ways.
>- **Mocks**: Mocks are objects pre-programmed with expectations which form a specification of the calls they are expected to receive.
>- **Stubs**: Stubs provide canned answers to calls made during the test, usually not responding to anything outside what's programmed in for the test.
>
> These tools help in isolating the code under test and ensuring that tests are not dependent on the behavior of external components.

## Running the Tests

To run the tests for the Aware-Microservices project, follow these steps:

1. Ensure that Docker Enginer is running.
2. Execute the following command to run the tests:

    ```bash
    python -m pytest tests/test_services_integration_with_db.py
    ```

This command will run the specified test suite using `pytest`.

Alternatively, you can run all the tests in the `tests` directory by executing:

```bash
pytest tests/
```

This command will run all the tests in the `tests` directory.