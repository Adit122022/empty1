import React, { useEffect, useState } from 'react';
import { MdDeleteForever } from "react-icons/md";
import { FaRegCheckCircle, FaRegCircle } from "react-icons/fa";
import Create from './Create';
import axios from 'axios';

const Home = () => {
  const [todos, setTodos] = useState([]);
  
  useEffect(() => {
    axios.get('http://localhost:3000/get')
      .then(result => setTodos(result.data))
      .catch(err => console.error(err.message));
  }, []);

  const handleEdit = (id) => {
    axios.put(`http://localhost:3000/update/${id}`)
      .then(result => location.reload())
      .catch(err => console.error(err.message));
  }

  const handleDelete = (id) => {
    axios.delete(`http://localhost:3000/delete/${id}`)
      .then(result => location.reload())
      .catch(err => console.error(err.message));
  }
  
  return (
    <div className="h-auto w-2/3 max-w-xl flex flex-col items-center justify-center mx-auto py-5 px-2 bg-gradient-to-r from-blue-500 via-indigo-600 to-purple-700 rounded-xl shadow-xl">
      <h2 className="text-4xl font-bold text-white mb-6 drop-shadow-lg">Todo List</h2>
      
      <div className="w-full max-w-md mx-auto">
        <Create />
      </div>
      
      {todos.length === 0 ? (
        <div className="text-center mt-10">
          <h2 className="text-xl font-semibold text-yellow-300">No Tasks Found</h2>
        </div>
      ) : (
        <div className="mt-6 w-full max-w-md space-y-6 h-[50vh] overflow-y-auto scroller">
          {todos.map((todo, index) => (
            <div
              key={index}
              className="bg-white shadow-lg rounded-lg px-4 py-2 flex justify-between items-center hover:shadow-2xl transition-all duration-300 ease-in-out transform hover:scale-105"
            >
              <div className="flex items-center gap-3" onClick={() => { handleEdit(todo._id) }}>
                <span className="flex items-center">
                  {todo.done ? <FaRegCheckCircle className='text-green-500 text-2xl' /> : <FaRegCircle className='text-red-400 text-2xl' />}
                </span>
                <span className={`text-lg font-medium ${todo.done ? 'line-through text-green-500' : 'text-gray-700'}`}>
                  {todo.task}
                </span>
              </div>

              <button
                onClick={() => { handleDelete(todo._id) }}
                className="text-black text-2xl p-2 rounded-md hover:bg-red-600 hover:text-white transition-all duration-300 active:scale-90"
              >
                <MdDeleteForever />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Home;
