import os
import tempfile

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile


class DataModel:
    """
    Model for handling data loading and processing.
    """

    @staticmethod
    def load_dataframe(uploaded_file: UploadedFile) -> tuple[pd.DataFrame | None, str | None]:
        """
        Load a dataframe from an uploaded file.

        Args:
            uploaded_file: The uploaded file object from streamlit

        Returns:
            A tuple of (pandas DataFrame, error message)
        """
        if not uploaded_file:
            return None, "No file uploaded"

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Read the file based on its extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == ".csv":
                df = pd.read_csv(tmp_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(tmp_path)
            else:
                return None, "Unsupported file format"

            # Clean up the temporary file
            os.unlink(tmp_path)

            return df, None
        except Exception as e:
            return None, str(e)
            
    @staticmethod
    def load_multiple_dataframes(
        uploaded_files: list[UploadedFile]
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """
        Load multiple dataframes from uploaded files with auto-generated table names.

        Args:
            uploaded_files: List of uploaded file objects

        Returns:
            A tuple of (dictionary of dataframes, list of error messages)
        """
        dataframes = {}
        errors = []
        # Track used table names to avoid duplicates
        used_names: set[str] = set()

        for file in uploaded_files:
            # Load the dataframe
            df, error = DataModel.load_dataframe(file)
            
            if error:
                errors.append(f"Error loading {file.name}: {error}")
                continue
            
            # Generate table name from filename (without extension)
            base_name = os.path.splitext(file.name)[0].lower().replace(" ", "_")
            
            # Handle duplicate names by adding version numbers
            table_name = base_name
            version = 1
            
            while table_name in used_names:
                # If name already exists, add version number
                table_name = f"{base_name}_v{version}"
                version += 1
            
            # Mark this name as used
            used_names.add(table_name)
                
            # Add to dictionary of dataframes
            dataframes[table_name] = df
            
        return dataframes, errors
