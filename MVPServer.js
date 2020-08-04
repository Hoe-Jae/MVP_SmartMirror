var fs = require('fs'), http = require('http'), express = require('express');
var url = require('url'), socket = require('socket.io');
var path = require('path'), serveStatic = require('serve-static');
var id = 0;
// HTTP Server //

var app = express();

app.use(express.static('public'));

app.use('/js', serveStatic(path.join(__dirname,'/js')));
app.use('/img', serveStatic(path.join(__dirname,'/img')));
app.use('/css', serveStatic(path.join(__dirname,'/css')));
app.use('/html', serveStatic(path.join(__dirname,'/html')));

var httpserver = http.createServer(app).listen(5318, function(){
  console.log('Server Running at http://127.0.0.1:5318');
  
});

app.get('/', function(request, response){
  fs.readFile('./html/Main.html', function(error, data){
    response.writeHead(200, {'Content-Type':'text/html'});
    response.end(data);
  });
});

app.get('/Active', function(request, response){
  response.writeHead(301, {'Location' : './html/Active'+request.query.ID+'.html'});
  response.end();
  
});



// Websocket Server //
var io = socket(httpserver);

io.on('connection', function(socket){
  console.log(socket.id);
});

// TCP Server //

var tcp = require('net');

var tcpserver = tcp.createServer(function(client){

  client.on('data', function(data){

    console.log('Client sent ' + data.toString());

    if(data.toString().includes('DATA')){
      id = data.toString().substr(4,5);
      io.emit('data', {ID : id});
    }

    if(data.toString()== 'CLOSE'){
      httpserver.close();
    }

    if(data.toString() == 'Home'){
      io.emit('home', {});
    }

  });

  client.on('end',function(){

    console.log('Client disconnected');


  });

  client.write('Hello Client!');

});

tcpserver.listen(5319, function(){
  console.log('Server Running at http://127.0.0.1:5319');
});


