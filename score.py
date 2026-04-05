from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from typer import Option

app = typer.Typer(no_args_is_help=True)


def rich_table(df, key='PES', title=''):
    df = df.round(1)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta", title=title)

    for col in df.columns:
        table.add_column(str(col))

    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row.values])

    console.print(table)


@app.command(help='Compare the results')
def compare(
        filename: Annotated[str, Option(help="CSV file name of results")] = "results.csv",
        save: Annotated[str, Option(help="Saving filtered results")] = None,
        task: Annotated[str, Option(help="Task name to filter")] = "reasoning",
):
    tasks_with_acc_norm = ['hellaswag', 'arc_challenge', 'mathqa', 'social_iqa', 'piqa', 'openbookqa']
    tasks_with_acc = ['arc_easy', 'boolq', 'mmlu', 'race', 'winogrande', 'commonsense_qa']

    tasks_with_gen = ['coqa', 'squadv2', 'triviaqa', 'nq_open', 'drop', 'gsm8k', 'truthfulqa_gen', 'xsum']

    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)

    if task.startswith('rea'):
        df = df[
            ((df['Task'].isin(tasks_with_acc_norm)) & (df['Metric'] == 'acc_norm')) |
            ((df['Task'].isin(tasks_with_acc)) & (df['Metric'] == 'acc'))
            ]
    elif task.startswith('gen'):
        df = df[
            ((df['Task'].isin(tasks_with_gen)) & (df['Metric'] == 'f1')) |
            ((df['Task'].isin(tasks_with_gen)) & (df['Metric'] == 'rouge')) |
            ((df['Task'].isin(tasks_with_gen)) & (df['Metric'] == 'rouge1_acc')) |
            ((df['Task'].isin(tasks_with_gen)) & (df['Metric'] == 'exact_match'))
            ]
        df = df.loc[df.groupby(["Model", "Task"])["Value"].idxmax()]

    # df = df[(~(df['Task'].isin(['boolq', 'arc_easy','arc_challenge'])))]
    # df = df[(~(df['Task'].isin(['boolq'])))]

    param = dict()
    for i, row in df[['Param(B)', 'Model']].iterrows():
        param[row['Model']] = float(row['Param(B)'])

    df = df.pivot(index='Model', columns='Task', values='Value')
    if task.startswith('gen'):
        try:
            df["squadv2"] = df["squadv2"] * 0.01
        except:
            pass
    df['average'] = df.mean(axis=1)
    df = df.round(3) * 100

    df = df.reset_index().rename(columns={'index': 'Model'})
    param_columns = [param[row['Model']] for _, row in df.iterrows()]
    df.insert(loc=0, column='Param(B)', value=param_columns)
    # df['Model'] = df["Model"].str.replace('llama2_7B_', '')

    df.sort_values(by=['average'], inplace=True)
    rich_table(df, title='AutoPrun')
    if save:
        df.to_csv(save, index=False)


if __name__ == '__main__':
    app()
