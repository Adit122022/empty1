import axios from 'axios';
import React, { useState } from 'react';

const Create = () => {
    const [task,setTask] = useState()
 
    const handleAdd=()=>{
axios.post("http://localhost:3000/add" ,{task:task}).then( result => location.reload()) .catch(err =>{console.   log(err)});
    }
  return (
    <div className="flex items-center gap-4 mb-6">
        <input 
          type="text"  onChange={(e)=>{ setTask(e.target.value)}}
          className="w-full max-w-xs p-2 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-purple-400 focus:outline-none" 
          placeholder="Enter a task..."
        />
        <button 
          type="button"  onClick={handleAdd}
          className="bg-blue-500 text-white px-4 py-2 rounded-lg shadow-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-300">
          Add Task
        </button>
    </div>
  );
};

export default Create;