from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    admin_token: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
