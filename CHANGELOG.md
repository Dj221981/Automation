# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-19

### Added
- Production GitHub Actions CI pipeline covering Python (3.9-3.11) and TypeScript jobs.
- Hardened developer tooling setup with `Makefile`, `pyproject.toml`, and `jest.config.js`.
- Containerization support with a multi-stage `Dockerfile`, `docker-compose.yml`, and `.dockerignore`.
- Repository automation templates for pull requests and issues.

### Changed
- Expanded `requirements.txt` with runtime, quality, testing, and security dependencies.
- Replaced root `README.md` with production-focused setup, configuration, and operations guidance.
