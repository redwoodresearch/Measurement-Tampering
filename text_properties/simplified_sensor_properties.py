import attrs

from text_properties.simplified_data_types import SimpleWritingResponse

@attrs.frozen
class :
    writing_response: SimpleWritingResponse
    sensor_values: dict[str, bool]

    def theoretical_tamper(self):
        return self.writing_response.full_response.theoretical_tamper()

    def has_latent(self):
        return self.writing_response.full_response.has_latent()
