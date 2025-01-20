const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const app = express()
const TodoModel =require('./src/Models/Todo')
app.use(cors())
app.use(express.json())

mongoose.connect('mongodb://localhost:27017/todo-list') .then(() => console.log('MongoDB Connected...')) .catch(err => console.log(err));
app.post('/add' , (req,res) =>{ 
    const task = req.body.task;
    TodoModel.create({task: task}) .then(result => res.json(result)).catch(err => console.log(err));    
})

app.get('/get' , (req, res) =>{
    TodoModel.find().then(result => res.json(result)).catch(err => res.json(err));
})
app.put('/update/:id', (req, res) => {
    const { id } = req.params;
    TodoModel.findById(id)
        .then(todo => {
            // Toggle the 'done' field
            todo.done = !todo.done;
            return todo.save();
        })
        .then(updatedTodo => res.json(updatedTodo))
        .catch(err => res.json(err.message));
});
app.delete('/delete/:id' , (req,res) =>{ 
    const {id} =req.params 
    TodoModel.findByIdAndDelete({_id:id}) .then(result=>res.json(result)) .catch(err=>res.json(err.message))
});
app.listen(3000 ,()=> { console.log("server listening on port 3000");
})