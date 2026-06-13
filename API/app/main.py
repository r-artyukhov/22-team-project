import time
from contextlib import asynccontextmanager
from datetime import timedelta

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_admin_user,
    get_password_hash,
    verify_admin_token,
)
from app.config import settings
from app.database import get_database_session, initialize_database
from app.log_parser import parse_logs
from app.ml_model import predict_from_logs
from app.models import RequestHistory, User
from app.prometheus_metrics import ANOMALY_PREDICTIONS, metrics_endpoint
from app.schemas import (
    AnomalyResponse,
    HistoryItem,
    HistoryResponse,
    LogSequenceRequest,
    StatsResponse,
    Token,
    UserCreate,
    UserResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await initialize_database()
    yield


app = FastAPI(title="ML Service API", version="2.0.0", lifespan=lifespan)


@app.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    user_data: UserCreate, session: AsyncSession = Depends(get_database_session)
):
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
    user = await authenticate_user(
        form_data.username, form_data.password, session
    )
    if not user:
        raise HTTPException(
            status_code=401, detail="Incorrect username or password"
        )
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.jwt_expiration_minutes),
    )
    return {"access_token": token, "token_type": "bearer"}


@app.post("/forward", response_model=AnomalyResponse)
async def forward(
    request: LogSequenceRequest,
    session: AsyncSession = Depends(get_database_session),
):
    start = time.time()
    result = predict_from_logs(parse_logs(request.logs))
    ANOMALY_PREDICTIONS.labels(
        "anomaly" if result["is_anomaly"] else "normal"
    ).inc()

    session.add(
        RequestHistory(
            request_type="log_anomaly_detection",
            processing_time=time.time() - start,
            input_data_size=len(request.logs),
            status_code=200,
            result=str(result),
        )
    )
    await session.commit()
    return AnomalyResponse(**result)


@app.get("/history", response_model=HistoryResponse)
async def get_request_history(
    _: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_database_session),
):
    items = (
        (
            await session.execute(
                select(RequestHistory).order_by(
                    RequestHistory.created_at.desc()
                )
            )
        )
        .scalars()
        .all()
    )
    return HistoryResponse(
        total=len(items), items=[HistoryItem.model_validate(i) for i in items]
    )


@app.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def delete_request_history(
    session: AsyncSession = Depends(get_database_session),
    _: bool = Depends(verify_admin_token),
):
    await session.execute(delete(RequestHistory))
    await session.commit()


@app.get("/stats", response_model=StatsResponse)
async def get_statistics(
    _: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_database_session),
):
    records = (await session.execute(select(RequestHistory))).scalars().all()
    times = [r.processing_time for r in records]
    sizes = [
        r.input_data_size for r in records if r.input_data_size is not None
    ]
    return StatsResponse(
        total_requests=len(records),
        mean_processing_time=float(np.mean(times)) if times else 0.0,
        median_processing_time=float(np.percentile(times, 50))
        if times
        else 0.0,
        percentile_95_processing_time=float(np.percentile(times, 95))
        if times
        else 0.0,
        percentile_99_processing_time=float(np.percentile(times, 99))
        if times
        else 0.0,
        average_input_size=float(np.mean(sizes)) if sizes else None,
    )


@app.get("/metrics")
async def prometheus_metrics():
    return metrics_endpoint()
