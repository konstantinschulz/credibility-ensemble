import json
from typing import Dict
from urllib.parse import quote
import requests
from docker import DockerClient

from config import Config
import time
import docker
from docker.models.containers import Container
import unittest


class ElgTestCase(unittest.TestCase):
    content: str = "Corona Virus, China & Shortage of Masks, Medicos. Behind the scenes, a small team of FBI agents spent years trying to solve a stubborn mystery — whether officials from Saudi Arabia, one of Washington’s closest allies, were involved in the worst terror attack in U.S. history."
    score: float = 1.0
    score_cs: float = 0.45345849802371535

    def test_cfc(self):
        client: DockerClient = docker.from_env()
        ports_dict: dict[int, int] = {Config.cfc_docker_port: Config.cfc_docker_port}
        container: Container = client.containers.run(Config.cfc_docker_image_name, ports=ports_dict,
                                                     detach=True)
        time.sleep(2)
        try:
            response: requests.Response = requests.get(Config.cfc_docker_base_url + quote(ElgTestCase.content))
            self.assertEqual(float(response.text), ElgTestCase.score)
        finally:
            container.stop()
            container.remove()
            client.close()

    def test_cs(self):
        client: DockerClient = docker.from_env()
        ports_dict: dict[int, int] = {Config.cs_docker_port: Config.cs_docker_port}
        container: Container = client.containers.run(Config.cs_docker_image_name, ports=ports_dict,
                                                     detach=True)
        time.sleep(1)
        try:
            response: requests.Response = requests.post(Config.cs_docker_base_url, json=dict(text=ElgTestCase.content))
            scores: Dict[str, float] = json.loads(response.text)
            self.assertEqual(scores["credibility_score_weighted"], ElgTestCase.score_cs)
        finally:
            container.stop()
            container.remove()
            client.close()

    # def test_local(self):
    #     request = TextRequest(content=ElgTestCase.content)
    #     service = CredibilityScoreService(Config.CREDIBILITY_SCORE_SERVICE)
    #     response = service.process_text(request)
    #     self.assertEqual(response.classes[0].score, ElgTestCase.score)
    #     self.assertEqual(type(response), ClassificationResponse)

    # def test_docker(self):
    #     client = docker.from_env()
    #     ports_dict: dict = dict()
    #     ports_dict[Config.DOCKER_PORT_CREDIBILITY] = Config.HOST_PORT_CREDIBILITY
    #     container: Container = client.containers.run(
    #         Config.DOCKER_IMAGE_CREDIBILITY_SERVICE, ports=ports_dict, detach=True)
    #     # wait for the container to start the API
    #     time.sleep(1)
    #     service: Service = Service.from_docker_image(
    #         Config.DOCKER_IMAGE_CREDIBILITY_SERVICE,
    #         f"http://localhost:{Config.DOCKER_PORT_CREDIBILITY}/process", Config.HOST_PORT_CREDIBILITY)
    #     response: Any = service(ElgTestCase.content, sync_mode=True)
    #     cr: ClassificationResponse = response
    #     container.stop()
    #     container.remove()
    #     self.assertEqual(cr.classes[0].score, ElgTestCase.score)
    #     self.assertEqual(type(response), ClassificationResponse)
    #
    # def test_elg_remote(self):
    #     service = Service.from_id(7348)
    #     response: Any = service(ElgTestCase.content)
    #     cr: ClassificationResponse = response
    #     self.assertEqual(cr.classes[0].score, ElgTestCase.score)
    #     self.assertEqual(type(response), ClassificationResponse)


if __name__ == '__main__':
    unittest.main()
