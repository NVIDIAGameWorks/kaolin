# Kaolin build commands for dash components

# Install all dependencies
install:
    npm install

# Build all dash components
build:
    npm run build:dash

# Build JavaScript bundles for dash components
build-js:
    npm run build:dash:js

# Generate Python backends for dash components
generate:
    npm run build:dash:backends

# Watch mode for development
watch:
    npm run build:dash:watch

# Clean build artifacts
clean:
    rm -rf dist
    rm -rf build
    rm -rf kaolin.egg-info
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} +

# Build Python package
package: clean build
    python setup.py build

# Install in development mode
dev-install: build
    pip install -e .

# Full development setup
dev-setup: install dev-install
