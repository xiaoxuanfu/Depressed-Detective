var token = '2038948207:AAHKguoaP89N9bivDZ5UtupQUN8JuWeetLs';
var telegramUrl = 'https://api.telegram.org/bot' + token;
var webAppUrl = 'https://script.google.com/macros/s/AKfycbwKDLPtigOhP2TmaukFno8DDM2iHHlthUVc1gDqJexJ8TeJQpU/exec';


function setWebhook() {
  var url = telegramUrl + '/setWebhook?url=' + webAppUrl;
  var response = UrlFetchApp.fetch(url);
}


function sendMessage(id, text, keyBoard){
  var data  = {
    method: 'post',
    payload: {
      method: 'sendMessage',
      chat_id: String(id),
      text: text ,
      parse_mode: 'HTML',
      reply_markup: JSON.stringify(keyBoard)
    }
  };
  UrlFetchApp.fetch('https://api.telegram.org/bot' + token + '/', data);
}


function doPost(e){
  var contents = JSON.parse(e.postData.contents);
  var ssId = '1mTY1z5O_A6Lu9Glq50huWMUh7IwEsiWMy6PjDaD0tfg';
  var sheet = SpreadsheetApp.openById(ssId).getSheetByName('depressed');
  var dateNow = new Date;
  var date_month = dateNow.getMonth() + 1
  var reformmatedDate = dateNow.getDate() + '/' + date_month;
  /*sheet.appendRow([reformmatedDate, text]);*/

if (contents.message){
    var id = contents.message.from.id;
    var text = contents.message.text;
    
    if (text == 'Start'){
      var keyBoard = {
      'inline_keyboard': [
        [{'text' : 'Month',
            'callback_data':'Month'
         }],
        [{'text' : 'Employees',
            'callback_data':'employees'
         }]
        ]
      };
      return sendMessage(id, 'Depressed detective mode on',keyBoard)
    }else {
      return sendMessage(id, 'Plese select the keyboard!')
    }
  }else if (contents.callback_query){
    var id = contents.callback_query.from.id;
    var data = contents.callback_query.data;
    var j = 1;
    if (data == 'Month') {
      var keyBoard2 = {
      'inline_keyboard': [
        [{'text' : 'January',
            'callback_data':'January'
         }],
        [{'text' : 'February',
            'callback_data':'February'
         }],
        [{'text' : 'March',
            'callback_data':'March'
         }],
        [{'text' : 'April',
            'callback_data':'April'
         }],
        [{'text' : 'May',
            'callback_data':'May'
         }],
        [{'text' : 'June',
            'callback_data':'June'
         }],
        [{'text' : 'July',
            'callback_data':'July'
         }],
        [{'text' : 'August',
            'callback_data':'August'
         }],
        [{'text' : 'September',
            'callback_data':'September'
         }],
        [{'text' : 'October',
            'callback_data':'October'
         }],
        [{'text' : 'November',
            'callback_data':'November'
         }],
        [{'text' : 'December',
            'callback_data':'December'
         }]
        ]
      };
      return sendMessage(id, 'Depressed detective mode on',keyBoard2);
    }else if (data == 'Employees') {
      sendMessage(id, 'Please enter the name.');
    }else {
      if (data == 'January'){
            j=2;
        
          }else if (data == 'February'){
            j=3;
          }
          else if (data == 'March'){
            j=4;
          }
          else if (data == 'April'){
            j=5;
          }
          else if (data == 'May'){
            j=6;
          }
          else if (data == 'June'){
            j=7;
          }else if (data == 'July'){
            j=8;
          }else if (data == 'August'){
            j=9;
          }else if (data == 'September'){
            j=10;
          }else if (data == 'October'){
            j=11;
          }else if (data == 'November'){
            j=12;
          }else if (data == 'December'){
            j=13;
          }else{
            return sendMessage(id, 'Please select the month.');
          }
        
    
      var namelist = 'Name list:';
      let count = 0;
      var stop = true;
      var i = 2;
      
      do{
        var person = sheet.getDataRange().getCell(i, 1).getValue();
        var dep = sheet.getDataRange().getCell(i, j).getValue();
        
        if (dep == '1'){
          namelist += ' ' + person ;
          count += 1;
        }else if (dep == '0'){
          
        }else {
          
          if (count == 0){
            sendMessage(id, 'Everyone is happy!');
          }else{
            sendMessage(id, namelist);
          }
          stop = false;
        }
        
        i += 1;
      }while (stop);
    }      
  }else {
  }
}