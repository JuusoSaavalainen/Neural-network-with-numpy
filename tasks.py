from invoke import task

@task
def startgui(ctx):
    ctx.run("python src/model/gui.py", pty=True)

@task
def train(ctx):
    ctx.run("python src/model/main.py", pty=True)
    
@task
def dowloadmnist(ctx):
    ctx.run("python src/data/idx_to_csv.py", pty=True)

