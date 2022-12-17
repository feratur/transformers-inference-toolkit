.PHONY:	lint format install clean build publish

# Run code quality checks
lint:
	isort --check .
	black --check .
	flake8 .

# Autoformat the code
format:
	isort .
	black .

# Install project dependencies
install:
	poetry install

# Delete previous build
clean:
	rm -rf dist

# Build the source and wheel archives
build: clean
	poetry build

# Publish the package to PyPI
publish: build
	poetry publish
