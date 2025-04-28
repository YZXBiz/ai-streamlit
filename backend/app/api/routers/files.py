"""File upload and management API endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.api.deps import get_current_active_user, get_file_service
from backend.app.api.schemas import DataFileResponse
from backend.app.core.security import TokenData
from backend.app.services.file_service import FileService

router = APIRouter(tags=["files"])


@router.post("", response_model=DataFileResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Form(""),
    current_user: TokenData = Depends(get_current_active_user),
    file_service: FileService = Depends(get_file_service),
) -> DataFileResponse:
    """
    Upload a data file.

    Accepts CSV, Excel, Parquet, or JSON files.
    """
    data_file = await file_service.upload_file(
        file=file, user_id=current_user.user_id, description=description
    )
    return DataFileResponse(
        id=data_file.id,
        user_id=data_file.user_id,
        filename=data_file.filename,
        original_filename=data_file.original_filename,
        file_path=data_file.file_path,
        file_size=data_file.file_size,
        file_type=data_file.file_type.name,
        description=data_file.description,
        created_at=data_file.created_at,
        updated_at=data_file.updated_at,
    )


@router.get("", response_model=list[DataFileResponse])
async def get_user_files(
    current_user: TokenData = Depends(get_current_active_user),
    file_service: FileService = Depends(get_file_service),
) -> list[DataFileResponse]:
    """Get all files for the current user."""
    data_files = await file_service.get_user_files(current_user.user_id)
    return [
        DataFileResponse(
            id=df.id,
            user_id=df.user_id,
            filename=df.filename,
            original_filename=df.original_filename,
            file_path=df.file_path,
            file_size=df.file_size,
            file_type=df.file_type.name,
            description=df.description,
            created_at=df.created_at,
            updated_at=df.updated_at,
        )
        for df in data_files
    ]


@router.get("/{file_id}", response_model=DataFileResponse)
async def get_file(
    file_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    file_service: FileService = Depends(get_file_service),
) -> DataFileResponse:
    """Get a specific file."""
    data_file = await file_service.get_file(file_id, current_user.user_id)
    if data_file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    return DataFileResponse(
        id=data_file.id,
        user_id=data_file.user_id,
        filename=data_file.filename,
        original_filename=data_file.original_filename,
        file_path=data_file.file_path,
        file_size=data_file.file_size,
        file_type=data_file.file_type.name,
        description=data_file.description,
        created_at=data_file.created_at,
        updated_at=data_file.updated_at,
    )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    file_service: FileService = Depends(get_file_service),
) -> None:
    """Delete a file."""
    await file_service.delete_file(file_id, current_user.user_id)
