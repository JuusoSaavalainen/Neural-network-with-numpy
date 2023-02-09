from invoke import task

@task
def startgui(ctx):
    ctx.run("python src/model/gui.py", pty=True)