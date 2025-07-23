# FastNext Starter API

A FastAPI backend for the Fast Next Starter project.

## Project Structure

```
api/
├── config/             # Configuration management
│   └── settings.py     # Application settings
├── models/             # Data models
│   ├── domain/         # Domain models
│   ├── schemas/        # Pydantic schemas
│   └── database/       # Database models
├── services/           # Business logic
├── utils/              # Utility functions
│   └── constants.py    # Application constants
├── core/              # Core functionality
├── tests/             # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── main.py            # Application entry point
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# Install all dependencies (including development tools)
pip install -r requirements.txt

# Or, if using pip-tools
pip install pip-tools
pip-compile pyproject.toml
pip-sync
```

3. Copy `.env.example` to `.env` and update the values:
```bash
cp .env.example .env
```

4. Run the development server:
```bash
uvicorn main:app --reload
```

## Development Tools

### Code Formatting (Black)
Black is our code formatter of choice, ensuring consistent code style.

```bash
# Format a specific file
black path/to/file.py

# Format the entire project
black .

# Check formatting without making changes
black . --check
```

Configuration: 100 character line length, targeting Python 3.9+

### Import Sorting (isort)
isort automatically organizes your imports according to PEP 8.

```bash
# Sort imports in a file
isort path/to/file.py

# Sort all project imports
isort .

# Check import sorting without making changes
isort . --check
```

Configuration: Black-compatible, 100 character line length

### Linting (Flake8)
Flake8 checks for style guide enforcement and common errors.

```bash
# Lint specific file
flake8 path/to/file.py

# Lint entire project
flake8 .
```

Common error codes:
- E### / W###: PEP 8 errors/warnings
- F###: PyFlakes errors
- C###: McCabe complexity

### Type Checking (MyPy)
MyPy performs static type checking of Python code.

```bash
# Type check specific file
mypy path/to/file.py

# Type check entire project
mypy .
```

Configuration:
- Strict type checking enabled
- Untyped definitions disallowed
- Return type checking enforced

### Testing (Pytest)
Pytest is our testing framework, supporting both sync and async tests.

```bash
# Run all tests
pytest

# Run specific test file
pytest path/to/test_file.py

# Run tests with coverage report
pytest --cov=app

# Run tests matching a pattern
pytest -k "test_pattern"
```

Test file naming convention: `test_*.py`

### Running All Checks

We provide a convenience script to run all checks in sequence:

```bash
# Format code
black . && isort .

# Run all checks
flake8 . && mypy . && pytest
```

## Environment Variables

### Setting Environment Variables

1. Create a `.env` file in the `api` directory
2. Add your variables in KEY=VALUE format:
```env
APP_NAME="My API"
DEBUG=true
```

### Adding New Environment Variables

1. Add the variable to `config/settings.py`:
```python
class Settings(BaseSettings):
    MY_NEW_VARIABLE: str = "default value"
```

2. Add it to your `.env` file:
```env
MY_NEW_VARIABLE="my value"
```

3. Use it in your code:
```python
from config.settings import get_settings

settings = get_settings()
value = settings.MY_NEW_VARIABLE
```

## Deployment

### Vercel Environment Variables

When deploying to Vercel, you need to configure the following environment variables:

#### Required Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `DEBUG` | Debug mode (should be false in production) | `false` |

#### Optional Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `APP_NAME` | Name of your API | `"FastNext Starter API"` |
| `APP_DESCRIPTION` | API description | `"A FastAPI backend for the Fast Next Starter project"` |
| `APP_VERSION` | API version | `"0.1.0"` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `https://your-domain.vercel.app,https://*.vercel.app` |

### Setting Up Environment Variables in Vercel

1. Go to your project settings in Vercel
2. Navigate to the "Environment Variables" section
3. Add each required variable
4. For preview deployments, you can set different values using the "Preview" environment

### Environment-Specific Configuration

- **Production**: Use your actual domain
  ```
  ENVIRONMENT=production
  DEBUG=false
  ALLOWED_ORIGINS=https://your-domain.vercel.app
  ```

- **Preview Deployments**: Allow Vercel preview URLs
  ```
  ENVIRONMENT=production
  DEBUG=false
  ALLOWED_ORIGINS=https://*.vercel.app
  ```

- **Development**: Local development settings
  ```
  ENVIRONMENT=development
  DEBUG=true
  ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
  ```

## API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Best Practices

1. **Code Quality Process**
   - Run formatters (Black, isort) before committing
   - Ensure all tests pass
   - Check type hints with MyPy
   - Keep test coverage high

2. **Git Workflow**
   - Create feature branches from main
   - Run all checks before pushing
   - Keep commits atomic and well-described

3. **Documentation**
   - Document all public functions and classes
   - Keep README up to date
   - Document API endpoints in code
