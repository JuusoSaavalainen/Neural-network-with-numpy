from invoke import task

@task
def startgui(ctx):
    ctx.run("python src/model/gui.py", pty=True)

@task
def train(ctx):
    ctx.run("python src/model/main.py", pty=True)