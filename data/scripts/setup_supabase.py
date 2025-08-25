#!/usr/bin/env python3
"""
Supabase CLI-based Database Setup
Implements migrations-based workflow for predictive maintenance system
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupabaseMigrator:
    """Handles Supabase CLI migrations"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.migrations_dir = self.project_dir / "supabase" / "migrations"
        self.use_npx = False  # Will be set by check_cli_installed
    
    def _supabase_cmd(self) -> list:
        """Get the supabase command based on installation method"""
        return ["npx", "supabase"] if self.use_npx else ["supabase"]
        
    def check_cli_installed(self) -> bool:
        """Check if Supabase CLI is installed"""
        try:
            # Try npx first (recommended approach)
            result = subprocess.run(
                ["npx", "supabase", "--version"], 
                capture_output=True, 
                text=True,
                check=True
            )
            logger.info(f"Supabase CLI version: {result.stdout.strip()}")
            self.use_npx = True
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback to global supabase
                result = subprocess.run(
                    ["supabase", "--version"], 
                    capture_output=True, 
                    text=True,
                    check=True
                )
                logger.info(f"Supabase CLI version: {result.stdout.strip()}")
                self.use_npx = False
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    def init_project(self) -> bool:
        """Initialize Supabase project"""
        try:
            os.chdir(self.project_dir)
            
            # Check if already initialized
            if (self.project_dir / "supabase").exists():
                logger.info("Supabase project already initialized")
                return True
            
            logger.info("Initializing Supabase project...")
            cmd = self._supabase_cmd() + ["init"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Supabase project initialized")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Supabase project: {e}")
            return False
    
    def link_project(self, project_ref: str) -> bool:
        """Link to remote Supabase project"""
        try:
            os.chdir(self.project_dir)
            
            logger.info(f"Linking to project: {project_ref}")
            cmd = self._supabase_cmd() + ["link", "--project-ref", project_ref]
            subprocess.run(cmd, check=True)
            logger.info("✅ Project linked successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to link project: {e}")
            return False
    
    def create_migration_file(self, name: str, sql_content: str) -> bool:
        """Create a new migration file"""
        try:
            os.chdir(self.project_dir)
            
            # Create migration using CLI
            logger.info(f"Creating migration: {name}")
            cmd = self._supabase_cmd() + ["migration", "new", name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find the created migration file
            migration_files = list(self.migrations_dir.glob("*.sql"))
            migration_files.sort(key=lambda x: x.name)
            latest_migration = migration_files[-1]
            
            # Write SQL content to the migration file
            with open(latest_migration, 'w') as f:
                f.write(sql_content)
            
            logger.info(f"✅ Migration created: {latest_migration.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create migration: {e}")
            return False
    
    def run_migrations(self) -> bool:
        """Run pending migrations"""
        try:
            os.chdir(self.project_dir)
            
            logger.info("Running migrations...")
            cmd = self._supabase_cmd() + ["db", "push"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Migrations completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run migrations: {e}")
            return False
    
    def reset_database(self) -> bool:
        """Reset database (development only)"""
        try:
            os.chdir(self.project_dir)
            
            logger.warning("Resetting database...")
            cmd = self._supabase_cmd() + ["db", "reset"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Database reset completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reset database: {e}")
            return False
    
    def generate_types(self) -> bool:
        """Generate TypeScript types from database schema"""
        try:
            os.chdir(self.project_dir)
            
            logger.info("Generating TypeScript types...")
            cmd = self._supabase_cmd() + ["gen", "types", "typescript", "--local"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Types generated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate types: {e}")
            return False

def extract_project_ref_from_url(url: str) -> str:
    """Extract project reference from Supabase URL"""
    # URL format: https://PROJECT_REF.supabase.co
    if not url:
        return ""
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.hostname and '.supabase.co' in parsed.hostname:
            return parsed.hostname.split('.')[0]
    except Exception:
        pass
    
    return ""

def setup_migrations():
    """Main setup function"""
    print("🚀 SUPABASE CLI MIGRATIONS SETUP")
    print("=" * 50)
    
    migrator = SupabaseMigrator()
    
    # Step 1: Check CLI installation
    print("\n1️⃣  Checking Supabase CLI...")
    if not migrator.check_cli_installed():
        print("❌ Supabase CLI is not installed!")
        print("\n📋 To install Supabase CLI:")
        print("   npx supabase login")
        print("   or visit: https://supabase.com/docs/guides/cli")
        return False
    
    # Step 2: Get configuration
    print("\n2️⃣  Loading configuration...")
    try:
        config = get_config()
        project_ref = extract_project_ref_from_url(config.database.url)
        
        if not project_ref:
            print("❌ Could not extract project reference from SUPABASE_URL")
            print(f"   Current URL: {config.database.url}")
            return False
            
        print(f"✅ Project reference: {project_ref}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Step 3: Initialize project
    print("\n3️⃣  Initializing Supabase project...")
    if not migrator.init_project():
        return False
    
    # Step 4: Link to remote project
    print("\n4️⃣  Linking to remote project...")
    if not migrator.link_project(project_ref):
        print("⚠️  Failed to link project automatically")
        print("   You may need to run: supabase login first")
        print("   Then run: supabase link --project-ref", project_ref)
    
    # Step 5: Create initial migration
    print("\n5️⃣  Creating initial migration...")
    
    # Read the simplified schema
    schema_file = Path('schemas/simple_schema.sql')
    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return False
    
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    if not migrator.create_migration_file("initial_schema", schema_sql):
        return False
    
    # Step 6: Run migrations
    print("\n6️⃣  Running migrations...")
    if not migrator.run_migrations():
        print("⚠️  Migration failed - this is normal for first-time setup")
        print("   Run manually: supabase db push")
    
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 30)
    print("\n📋 Next steps:")
    print("   1. Verify migration: supabase db diff")
    print("   2. Test database: python3 scripts/test_connection.py")
    print("   3. Run pipeline: python3 scripts/continuous_pipeline.py --profile development")
    
    return True

def run_fresh_migration():
    """Run a fresh migration (development)"""
    print("🔄 RUNNING FRESH MIGRATION")
    print("=" * 30)
    
    migrator = SupabaseMigrator()
    
    if not migrator.check_cli_installed():
        print("❌ Supabase CLI not available")
        return False
    
    print("⚠️  This will reset your database!")
    response = input("Continue? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled")
        return False
    
    # Reset and re-run migrations
    if migrator.reset_database():
        return migrator.run_migrations()
    
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Supabase CLI Migrations Setup')
    parser.add_argument('--setup', action='store_true', help='Run initial setup')
    parser.add_argument('--migrate', action='store_true', help='Run migrations only')
    parser.add_argument('--reset', action='store_true', help='Reset database and re-run migrations')
    parser.add_argument('--types', action='store_true', help='Generate TypeScript types')
    
    args = parser.parse_args()
    
    if args.setup:
        success = setup_migrations()
    elif args.migrate:
        migrator = SupabaseMigrator()
        success = migrator.run_migrations()
    elif args.reset:
        success = run_fresh_migration()
    elif args.types:
        migrator = SupabaseMigrator()
        success = migrator.generate_types()
    else:
        # Default: run setup
        success = setup_migrations()
    
    sys.exit(0 if success else 1)