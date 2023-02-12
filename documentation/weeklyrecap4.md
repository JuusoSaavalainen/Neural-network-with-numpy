# Weekly Report 4



## Implementing GUI and CLI, model fixes



This week contained lots of experimenting with the model. While doing so it became clear that batching should be used to reduce the training time. Also noticed some mistakes in the math. Little improvements with batching yielded the best results yet. With the working model, I decided to implement GUI to it and for the training MVP of cli was made. One of the goals set for this week was testing the main algorithms. Many different variations of ways to test the forward prop were considered, and most ended up testing nothing relevant. The usage of seed() is a good way to implement relevant tests. This week I learned most about the variations of the methods of the model and how tuning parameters affect the outcome. 

## Next Week

- MAIN GOAL: Produce meaningful tests to main algorithms

- Clean code, especially with GUI. Cli is mvp so adding error expectations at least should be done next week

- Comparison between different models and times with plots could be done also next week

