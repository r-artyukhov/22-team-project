import time
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from datetime import timedelta

from app.config import settings
from app.database import get_database_session
from app.models import RequestHistory, User
from app.schemas import (
    HistoryResponse,
    HistoryItem,
    StatsResponse,
    UserCreate,
    UserResponse,
    Token,
    LogSequenceRequest,
    AnomalyResponse,
)
from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_admin_user,
    get_password_hash,
    verify_admin_token,
)
from app.ml_model import get_ml_model

app = FastAPI(title="ML Service API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "bad request"})


@app.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    user_data: UserCreate, session: AsyncSession = Depends(get_database_session)
):
    existing_user = await session.execute(select(User).where(User.username == user_data.username))
    if existing_user.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    new_user = User(
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        is_admin=user_data.is_admin,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return new_user


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_database_session),
):
    user = await authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.jwt_expiration_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/forward", response_model=AnomalyResponse)
async def forward(
    request: LogSequenceRequest,
    session: AsyncSession = Depends(get_database_session),
):
    """Детекция аномалий в последовательности логов."""
    start_time = time.time()

    if not isinstance(request.logs, list) or len(request.logs) == 0:
        raise HTTPException(status_code=400, detail="bad request")
    try:
        model = get_ml_model()
        logs_data = [log.model_dump() for log in request.logs]
        result = model.predict_from_logs(logs_data)

        processing_time = time.time() - start_time

        history_record = RequestHistory(
            request_type="log_anomaly_detection",
            processing_time=processing_time,
            input_data_size=len(logs_data),
            status_code=200,
            result=str(result),
        )
        session.add(history_record)
        await session.commit()

        return AnomalyResponse(
            score=result["score"],
            is_anomaly=result["is_anomaly"],
            threshold=result["threshold"],
            num_events=result["num_events"],
        )

    except Exception as e:
        processing_time = time.time() - start_time
        history_record = RequestHistory(
            request_type="log_anomaly_detection",
            processing_time=processing_time,
            input_data_size=len(request.logs),
            status_code=403,
            error_message="модель не смогла обработать данные",
        )
        session.add(history_record)
        await session.commit()
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")


@app.get("/history", response_model=HistoryResponse)
async def get_request_history(
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_database_session),
):
    result = await session.execute(
        select(RequestHistory).order_by(RequestHistory.created_at.desc())
    )
    history_items = result.scalars().all()

    return HistoryResponse(
        total=len(history_items),
        items=[HistoryItem.model_validate(item) for item in history_items],
    )


@app.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def delete_request_history(
    session: AsyncSession = Depends(get_database_session),
    verified: bool = Depends(verify_admin_token),
):
    await session.execute(delete(RequestHistory))
    await session.commit()
    return None


@app.get("/stats", response_model=StatsResponse)
async def get_statistics(
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_database_session),
):
    result = await session.execute(select(RequestHistory))
    all_records = result.scalars().all()

    if not all_records:
        return StatsResponse(
            total_requests=0,
            mean_processing_time=0.0,
            median_processing_time=0.0,
            percentile_95_processing_time=0.0,
            percentile_99_processing_time=0.0,
            average_input_size=None,
        )

    processing_times = [record.processing_time for record in all_records]
    input_sizes = [
        record.input_data_size for record in all_records if record.input_data_size is not None
    ]

    mean_time = float(np.mean(processing_times))
    median_time = float(np.percentile(processing_times, 50))
    p95_time = float(np.percentile(processing_times, 95))
    p99_time = float(np.percentile(processing_times, 99))

    avg_input_size = float(np.mean(input_sizes)) if input_sizes else None

    return StatsResponse(
        total_requests=len(all_records),
        mean_processing_time=mean_time,
        median_processing_time=median_time,
        percentile_95_processing_time=p95_time,
        percentile_99_processing_time=p99_time,
        average_input_size=avg_input_size,
    )


@app.get("/")
async def root():
    return {
        "message": "ML Service API - Log Anomaly Detection",
        "version": "1.0.0",
        "endpoints": {
            "POST /register": "Register a new user",
            "POST /token": "Get JWT access token",
            "POST /forward": "Detect anomalies in log sequence (Isolation Forest)",
            "GET /history": "Get request history (admin only)",
            "DELETE /history": "Delete request history (requires admin token)",
            "GET /stats": "Get statistics (admin only)",
        },
    }
