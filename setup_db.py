#!/usr/bin/env python3
"""
Database setup script.
Creates the database if it doesn't exist and runs the schema.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import psycopg
from psycopg import sql


def get_db_name() -> str:
    """Get the database name from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Parse DATABASE_URL: postgresql://user:password@host:port/dbname
        # For simplicity, we'll require DB_NAME to be set if using DATABASE_URL
        db_name = os.getenv("DB_NAME")
        if not db_name:
            print("Error: DB_NAME must be set when using DATABASE_URL")
            sys.exit(1)
        return db_name
    
    db_name = os.getenv("DB_NAME")
    if not db_name:
        print("Error: DB_NAME environment variable is not set")
        print("Please set DB_NAME in your .env file or environment")
        sys.exit(1)
    return db_name


def get_postgres_connection_kwargs() -> dict:
    """Get connection kwargs for connecting to the default 'postgres' database."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Parse and modify DATABASE_URL to connect to 'postgres' database
        # This is a simple approach - in production you might want more robust parsing
        db_name = os.getenv("DB_NAME")
        if db_name and f"/{db_name}" in database_url:
            # Replace the database name with 'postgres'
            modified_url = database_url.replace(f"/{db_name}", "/postgres")
            return {"conninfo": modified_url}
        # If we can't parse it, try connecting to postgres directly
        return {"conninfo": database_url.rsplit("/", 1)[0] + "/postgres"}
    
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": "postgres",  # Connect to default postgres database
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def database_exists(conn: psycopg.Connection, db_name: str) -> bool:
    """Check if a database exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        return cur.fetchone() is not None


def create_database(db_name: str):
    """Create the database if it doesn't exist."""
    print(f"Connecting to PostgreSQL server...")
    try:
        kwargs = get_postgres_connection_kwargs()
        with psycopg.connect(**kwargs, autocommit=True) as conn:
            if database_exists(conn, db_name):
                print(f"Database '{db_name}' already exists.")
                return
            
            print(f"Creating database '{db_name}'...")
            # Note: CREATE DATABASE cannot be run in a transaction
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(db_name)
                ))
            print(f"Database '{db_name}' created successfully.")
    except psycopg.Error as e:
        print(f"Error creating database: {e}")
        sys.exit(1)


def run_schema(db_name: str):
    """Run the schema.sql file to create tables."""
    schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")
    
    if not os.path.exists(schema_file):
        print(f"Error: schema.sql not found at {schema_file}")
        sys.exit(1)
    
    print(f"Reading schema from {schema_file}...")
    with open(schema_file, "r") as f:
        schema_sql = f.read()
    
    print(f"Applying schema to database '{db_name}'...")
    try:
        from db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
        print("Schema applied successfully.")
    except Exception as e:
        print(f"Error applying schema: {e}")
        sys.exit(1)


def main():
    print("Setting up database...")
    db_name = get_db_name()
    print(f"Database name: {db_name}")
    
    create_database(db_name)
    run_schema(db_name)
    
    print("\nDatabase setup complete!")


if __name__ == "__main__":
    main()
