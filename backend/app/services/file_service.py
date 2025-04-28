"""
File service for handling file operations.

This service handles file uploads, downloads, and related operations
including preprocessing and registering files with the data analysis system.
"""

import os
from typing import Any

from fastapi import HTTPException, UploadFile, status

from ..domain.models.datafile import DataFile, FileType
from ..ports.llm import DataAnalysisService
from ..ports.repository import DataFileRepository
from ..ports.storage import FileStorage


class FileService:
    """
    Service for handling file uploads, storage, and loading into dataframes.

    This service sits between the API layer and the storage/repository layers,
    coordinating the various operations needed for file management.
    """

    def __init__(
        self,
        file_repository: DataFileRepository,
        file_storage: FileStorage,
        data_analysis_service: DataAnalysisService,
    ):
        """
        Initialize the file service with its required dependencies.

        Args:
            file_repository: Repository for file metadata
            file_storage: Storage for file contents
            data_analysis_service: Service for analyzing data
        """
        self.file_repository = file_repository
        self.file_storage = file_storage
        self.data_analysis_service = data_analysis_service

    async def upload_file(self, file: UploadFile, user_id: int, description: str = "") -> DataFile:
        """
        Upload a file and store it.

        Args:
            file: The file to upload
            user_id: ID of the user uploading the file
            description: Optional description of the file

        Returns:
            DataFile: The created file record

        Raises:
            HTTPException: If the file format is not supported or if an error occurs
        """
        # Get file extension and determine file type
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()

        # Determine file type from extension
        if file_ext == ".csv":
            file_type = FileType.CSV
        elif file_ext in [".xlsx", ".xls"]:
            file_type = FileType.EXCEL
        elif file_ext == ".parquet":
            file_type = FileType.PARQUET
        elif file_ext == ".json":
            file_type = FileType.JSON
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}. Supported formats: .csv, .xlsx, .xls, .parquet, .json",
            )

        try:
            # Read file content
            file_content = await file.read()

            # Store file in storage
            filename = await self.file_storage.save_file(
                file_content, user_id, original_filename=file.filename
            )

            # Create file record in database
            file_record = DataFile(
                user_id=user_id,
                filename=filename,
                original_filename=file.filename,
                file_type=file_type,
                description=description,
            )

            created_file = await self.file_repository.create(file_record)
            return created_file

        except Exception as e:
            # Clean up - if file was saved to storage but db insert failed
            if "filename" in locals():
                await self.file_storage.delete_file(filename, user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File upload failed: {str(e)}",
            ) from e

    async def load_dataframe(self, file_id: int, user_id: int, name: str = "") -> Any:
        """
        Load a file into a dataframe for analysis.

        Args:
            file_id: ID of the file to load
            user_id: ID of the user requesting the file
            name: Optional name for the dataframe

        Returns:
            Any: The loaded dataframe (usually a pandas.DataFrame)

        Raises:
            HTTPException: If the file is not found or cannot be loaded
        """
        # Get file record from database
        file_record = await self.file_repository.get(file_id)

        if file_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found",
            )

        # Check if user has access to this file
        if file_record.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this file",
            )

        # Get file path from storage
        file_path = await self.file_storage.get_file_path(file_record.filename, user_id)

        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File content not found for file ID {file_id}",
            )

        # Use a name for the dataframe if provided, otherwise use original filename
        df_name = name if name else os.path.splitext(file_record.original_filename)[0]

        # Load dataframe using data analysis service
        try:
            dataframe = await self.data_analysis_service.load_dataframe(
                file_path, df_name, file_record.description
            )
            return dataframe
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load dataframe: {str(e)}",
            ) from e

    async def get_user_files(self, user_id: int) -> list[DataFile]:
        """
        Get all files for a user.

        Args:
            user_id: ID of the user

        Returns:
            list[DataFile]: List of file records
        """
        return await self.file_repository.get_by_user(user_id)

    async def get_file(self, file_id: int, user_id: int) -> DataFile:
        """
        Get a specific file.

        Args:
            file_id: ID of the file
            user_id: ID of the user requesting the file

        Returns:
            DataFile: The file record

        Raises:
            HTTPException: If the file is not found or user doesn't have access
        """
        file_record = await self.file_repository.get(file_id)

        if file_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found",
            )

        # Check if user has access to this file
        if file_record.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this file",
            )

        return file_record
