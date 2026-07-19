# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-19

### Added
- Production CI workflow for Python and TypeScript validation on pushes and pull requests.
- Comprehensive dependency management for runtime, linting, testing, and security tooling.
- Standardized local developer workflows through a root `Makefile`.
- Containerization assets including multi-stage `Dockerfile`, `docker-compose.yml`, and `.dockerignore`.
- Unified Python tooling configuration in `pyproject.toml`.
- Jest configuration for TypeScript test execution.
- GitHub collaboration templates for pull requests, bug reports, and feature requests.

### Changed
- Replaced the repository README with production-grade setup, structure, configuration, and Docker guidance.
