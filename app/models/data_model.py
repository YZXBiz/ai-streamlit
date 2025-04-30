import os
import tempfile

import pandas as pd


class DataModel:
    """
    Model for handling data loading and processing.
    """

    @staticmethod
    def load_dataframe(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
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
