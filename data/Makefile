# Predictive Maintenance System - Makefile
# Supabase CLI Migration Workflow

.PHONY: help setup migrate test reset clean install

# Default target
help:
	@echo "🚀 Predictive Maintenance System - Supabase Migrations"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make setup      - Initial Supabase setup with migrations"
	@echo "  make migrate    - Run pending migrations"
	@echo "  make test       - Test database connection and tables"
	@echo "  make reset      - Reset database and re-run migrations (dev only)"
	@echo "  make clean      - Clean up temporary files"
	@echo "  make pipeline   - Run the data pipeline (development)"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Initial setup
setup:
	@echo "🚀 Running initial Supabase setup..."
	python3 scripts/setup_supabase.py --setup

# Run migrations
migrate:
	@echo "🔄 Running migrations..."
	python3 scripts/setup_supabase.py --migrate

# Test connection and tables
test:
	@echo "🧪 Testing database connection..."
	python3 scripts/test_connection.py

# Reset database (development only)
reset:
	@echo "⚠️  Resetting database..."
	python3 scripts/setup_supabase.py --reset

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete

# Run development pipeline
pipeline:
	@echo "🔧 Running development pipeline..."
	python3 scripts/continuous_pipeline.py --profile development --duration 60

# Full workflow: setup -> test -> run pipeline
full:
	@echo "🚀 Running full workflow..."
	make setup
	make test
	make pipeline

# Development workflow
dev: install setup test
	@echo "✅ Development environment ready!"
	@echo "Run 'make pipeline' to start the data pipeline"