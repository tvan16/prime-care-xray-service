"""Xác thực JWT (HS512) do user-service phát hành và chỉ cho bác sĩ dùng."""
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from app.config import get_settings


@dataclass(frozen=True)
class CurrentUser:
    user_id: str  # claim `sub` = uuid của user
    role: Optional[str]


def decode_token(token: str, secret: str) -> CurrentUser:
    if len(secret.encode("utf-8")) < 64:
        # user-service cũng yêu cầu >= 64 byte cho HS512
        raise HTTPException(status_code=503, detail="Service chưa được cấu hình JWT_SECRET_KEY hợp lệ")
    try:
        payload = jwt.decode(
            token,
            secret.encode("utf-8"),
            algorithms=["HS512"],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn")
    return CurrentUser(user_id=str(payload["sub"]), role=payload.get("role"))


async def require_doctor(authorization: Optional[str] = Header(default=None)) -> CurrentUser:
    """Chốt chặn thật: gateway chỉ kiểm tra token hợp lệ, không kiểm tra vai trò."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Vui lòng đăng nhập")
    user = decode_token(authorization[7:].strip(), get_settings().jwt_secret_key)
    if user.role not in get_settings().allowed_roles:
        raise HTTPException(status_code=403, detail="Chỉ bác sĩ mới được sử dụng tính năng phân tích phim X-quang")
    return user
