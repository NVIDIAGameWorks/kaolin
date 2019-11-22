from kaolin.rep.TriangleMesh import TriangleMesh

def test_homo_obj():
    TriangleMesh.from_obj('tests/model.obj')

def test_hetero_obj():
    TriangleMesh.from_obj('tests/rep/shuttle.obj')