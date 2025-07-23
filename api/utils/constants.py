from enum import Enum


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# HTTP Response Messages
class ResponseMessage:
    SUCCESS = "Success"
    NOT_FOUND = "Resource not found"
    UNAUTHORIZED = "Unauthorized access"
    FORBIDDEN = "Access forbidden"
    VALIDATION_ERROR = "Validation error"
    INTERNAL_ERROR = "Internal server error"


# Common Status Codes
class StatusCode:
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_500_INTERNAL_SERVER_ERROR = 500 