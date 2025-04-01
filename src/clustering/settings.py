"""Define job settings for the application."""

# %% IMPORTS

import pydantic as pdt
from pydantic_settings import BaseSettings

from clustering import jobs


# %% SETTINGS
class MainSettings(BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Main settings of the application.

    Parameters:
        job (jobs.JobKind): job to run.
    """

    job: jobs.JobKind = pdt.Field(..., discriminator="KIND")
