from providers.base import GenerationRequest, Provider


class LiteratureAgent:
    name = "literature"

    def __init__(self, provider: Provider):
        self.provider = provider

    def run(self, mission: dict, candidate: dict) -> dict:
        return self.provider.generate(GenerationRequest(self.name, mission, {"candidate": candidate}))

