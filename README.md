# Automation Platform

[![CI](https://github.com/Dj221981/Automation/actions/workflows/ci.yml/badge.svg)](https://github.com/Dj221981/Automation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Dj221981/Automation)](LICENSE)
[![Coverage](https://codecov.io/gh/Dj221981/Automation/branch/main/graph/badge.svg)](https://codecov.io/gh/Dj221981/Automation)

Automation is a hybrid Python/TypeScript platform for reinforcement-learning automation workflows, featuring DQN training components and agent orchestration.

## Quick Start

```bash
git clone https://github.com/Dj221981/Automation.git
cd Automation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -v --cov=src
```

## Project Structure

| Path | Purpose |
| --- | --- |
| `/src` | Core Python services, models, and agents |
| `/tests` | Python and TypeScript test suites |
| `/storage` | TypeScript storage layer assets |
| `/config` | Runtime configuration files, including `production.json` |
| `DEVELOPMENT.md` | Developer workflow and contribution guide |

## Configuration

- Use `config/production.json` as the baseline production configuration.
- Copy `.env.example` to `.env` and set environment-specific values.
- Keep secrets in environment variables and never commit them.

## Docker

```bash
docker build -t automation:latest .
docker compose up --build
```

The compose stack runs the application container and a TensorBoard service on port `6006`.

## Contributing

Please follow the standards and workflow in [DEVELOPMENT.md](DEVELOPMENT.md) before opening a pull request.

## License

This project is licensed under the MIT License.
