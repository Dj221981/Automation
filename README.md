# Automation

![CI](https://github.com/Dj221981/Automation/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-CodeCov-orange)

Automation is a hybrid Python/TypeScript automation platform with a DQN reinforcement-learning core and super-agentic orchestration components designed for production deployment.

## Quick Start

```bash
git clone https://github.com/Dj221981/Automation.git
cd Automation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -v --cov=src
```

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/` | Python application logic, DQN model, training and agent orchestration |
| `tests/` | Python and TypeScript test suites |
| `config/production.json` | Production configuration defaults |
| `storage/` | TypeScript storage-related modules and assets |
| `.github/workflows/ci.yml` | CI pipeline for Python and TypeScript validation |

## Configuration

- Use `config/production.json` as the production baseline.
- Copy `.env.example` to `.env` and override environment-specific values.
- Keep secrets out of source control and inject them via environment variables.

## Docker

```bash
docker build -t automation:latest .
docker compose up --build
```

The compose stack includes the main app container and a TensorBoard service on `http://localhost:6006`.

## Contributing

Please follow `DEVELOPMENT.md` for setup, coding standards, and contribution workflow.

## License

This project is licensed under the MIT License.
