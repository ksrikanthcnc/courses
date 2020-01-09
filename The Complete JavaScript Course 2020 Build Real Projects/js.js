console.log("Hello world");

// number string boolean undefined(not set yet) null(set to null/deleting)
var variable;
// also checks types before/without casting
1 === "1"; //false
1 !== "1"; //true

function sum(a, b) {
  return a + b;
}
sum(1, 2);

var summer = function(a, b) {
  return a + b;
};
summer(1, 2);

arr = [1, "2"];
arr.length;
arr.push("end"); //add to last
arr.unshift("first"); //add to first
arr.pop(); //remove last element
arr.shift(); //remove first element
arr.indexOf(1);

// object
var john = {
  firstName: "John",
  age: 23,
  getAge: function(a) {
    console.log(a);
    return this.age;
  }
};
john.newProp = "new";
john.firstName;
john["firstName"];
john.getAge();

// Hoisting
// var (set to undefined) and functions are first parsed before execution

// var is scoped to function    (normal, nothing new)

window; // is the object of the whole window
document; // is the document
document.getElementById("id");
document.querySelector("#id").innerHTML = "new text";
document.title = "new title";
document.querySelector(".btn").addEventListener("click", func); //func should be defined here or else where

// only properties,attributes in 'prototype' are inherited
// constructor
var Person = function(name, age) {
  this.name = name;
  this.age = age;
  // each has its own opy of function
  this.getAge = function() {
    return this.age;
  };
};
// common for all objects
Person.prototype.getName = function() {
  return this.name;
};
Person.prototype.kind = "human";
var john = new Person("John", 23);
var mark = new Person("mark", 50);

john.hasOwnProperty("name"); //true; false for kind(because not own, but inherited)
john instanceof Person;

//Object.create (first creates prototype, then creates objects)
var personProto = {
  getAge: function() {
    return this.age;
  }
};
var john = Object.create(personProto);
john.age = 23;
var mark = Object.create(personProto, {
  name: { value: "John" }
});

// All objets are references

// First Class Functions
function func(a) {
  return a + 1;
}
function applyfn(num, fn) {
  return fn(num);
}
applyfn(1, func);

function inc(num) {
  return num + 1;
}
function dec(num) {
  return num - 1;
}
function retfn(ele) {
  if (ele >= 0) return inc;
  else return dec;
}
var resultfn = retfn(1);
resultfn(5);
var resultfn = retfn(-1);
resultfn(-5);
retfn(-1)(-5);

// IIFE (python's lambda)      For scope/... reasons
var r =
  (function(a) {
    var b = a;
    console.log(b);
    return b;
  })(5) + 1;

// Closures
// Local vars linger even if function is not in execution
function outer(arg) {
  var fvar = arg;
  return function inner() {
    return fvar + 1;
  };
}
var val = outer(0);
val();
var val2 = outer(5);
val2();

// Bind Call Apply
var john = {
  name: "John",
  getName: function(arg) {
    console.log(arg);
    return this.name;
  }
};
john.getName();
var emily = { name: "Emily" };
john.getName.call(emily, "new"); //  using john's method for emily
john.getName.apply(emily, ["new"]); //  second arg must be array
var bound = john.getName.bind(emily); // carrying(preset parameters, limit/restrict arguments)
bound("now pass arg");

var module = (function() {
  var public;
  var halfprivate;
  var halfprivatefn = function(arg) {
    return halfprivate + arg;
  };
  var fullprivate;
  var fullprivatefn = function(arg) {
    return fullprivate;
  };

  return {
    public1: public,
    public2: function() {
      return halfprivatefn;
    }
  };
})();

// ES6
let variable = 6; // scoped to block, var is scoped to whole(even out of block) function(or global)
const pi = 3.14;
str = `${variable} and ${pi}`;

const arr = [1, 2, 3];
arr.map(function(e, index) {
  return e + 1;
});
arr.map((e, index) => {
  return e + 1;
});

// DeStructuring
var john = ["John", 23];
var [name, age] = john;

var john = { fN: "John", age: 23 };
const { fN, age } = john; //should match keynames
const { fN: N, age: A } = john;

Array.from(arr).forEach(ele => console.log(ele));
for (ele in arr) console.log(ele);
arr.findIndex(ele => ele % 3 == 0);

// Spread
var arr = [0, 1, 2];
function sum(a, b, c) {
  return a + b + c;
}
var sum5 = sum.apply(null, arr);
var sum6 = sum(...arr);

// variable number of args to funtion
var arg5 = Array.prototype.slice.call(args); //slices aa args
var arg6 = [...args]; //function(...args){<use as args>}

// Default parameters

// Maps
var map = new Map();
map.set("key", "value");
map.get("key");
map.forEach((value, key) => console.log(key + value));
map.entries();
map.delete("key");
map.clear();

// Classes  (not hoisted)
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  getAge() {
    //public
    return this.age;
  }
  static greet() {
    console.log("Hello");
  }
}

// Sub Class
var subPerson5 = function(oldattr, newattr) {
  subPerson5.call(this, oldattr);
  this.newattr = newattr;
};
subPerson5.prototype = Object.create(Person5.prototype);
subPerson5.prototype.newMethod = function() {
  console.log("new only in sub");
};

class subPerson6 extends Person6 {
  constructor(oldattr, newattr) {
    super(oldattr);
    this.newattr = newattr;
  }
  newMethod = function() {
    console.log("new only in sub");
  };
}

// Asynchronous
var asyncfn = setTimeout(() => {
  console.log("2s lapsed");
}, 2000);
// ES5 nested asyn - callback hell
// ES6 promise
const prom = new Promise((resolve, reject) => {
  setTimeout(args => {
    console.log(args);
    if (success) resolve(result);
    if (failed) reject(errorCode);
  }, 2000);
});
const innerProm = innerArgs => {
  return new Promise((resolve, reject) => {
    setTimeout(innerArgs => {
      console.log(innerArgs);
      if (success) resolve(innerResult);
      if (failed) reject(innerErrorCode);
    }, 2000);
  });
};
// this will start execution, guess
prom.then(result => {
  console.log(result);
  return innerProm(innerArgs);
});
prom.then(innerResult => {
  console.log(innerResult);
  return finalResult;
});
prom.catch(errorCode => {
  console.log(errorCode);
});
// can also oncatenate prom.then().catch()
// async to ease consuming promises
async function getFinal() {
  try {
    var firstResult = await prom; //resolve
  } catch (error) {
    handle(error); // reject
  }
  console.log(firstResult);
  var secondresult = await result;
  console.log(secondresult);
  // cant return directly
  return finalResult;
}
getFinal().then(finalResult => console.log(finalResult));

// AJAX
fetch("URL")
  .then(result => {
    //usually the webpage
    return args;
  })
  .then(args => {
    //maybe some JSON object
    var finalResult = args.json();
    return finalResult;
  });
async function getPage() {
  var result = await fetch("URL");
  const finalResult = await finalResult.json();
  return ret;
}

// Node.js
const http = require("http");
const url = require("url");
const server = http.createServer((req, res) => {
  const path = url.parse(req.url, true).pathname;
  const query = url.parse(req.url, true).query; //JSON object with queries/parameters
  res.writeHead(200, { "Content-type": "text/html" });
  res.end("This is " + path + query);
});
