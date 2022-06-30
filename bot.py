#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
#import pickledb
from sympy import *
import numpy as np
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


MOD, OBT, FUNC, START, N_ITER = range(5)

PORT = int(os.environ.get('PORT', '8443'))

i_model = ''
i_obt = ''
i_func = ''
i_start = 0
i_niter = 1


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def choose_opt(update: Update, context: CallbackContext) -> int:
    """Choose Optimization """
    reply_keyboard = [['Gradient', 'NewtonMult', 'NewtonUni', 'Bisection']]

    update.message.reply_text(
        'Insert optimization model',
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),
    )

    return MOD


def obt(update: Update, context: CallbackContext) -> int:
    """Starts the conversation """
    reply_keyboard = [['max', 'min']]
    i_model = update.message.text

    context.user_data["model"] = i_model
    update.message.reply_text(
        'Do you want to max or min?',
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),
    )

    return OBT


def func(update: Update, context: CallbackContext) -> int:
    """Get objective"""

    #reply_keyboard = [['x', 'y', 'z', '^'],['7','8','9','*'],['4','5','6','/'], [ '1', '2', '3','+'], ['0','-']]
    i_obt = update.message.text
    context.user_data["obt"] = i_obt
    update.message.reply_text(
        'Insert the function',
        reply_markup=ReplyKeyboardRemove()
    )

    return FUNC


def start_point(update: Update, context: CallbackContext) -> int:
    """Get function."""

    #reply_keyboard = [['7','8','9'],['4','5','6'], [ '1', '2', '3'], ['0','-',',']]
    i_func = update.message.text
    i_func = i_func.replace("^", "**")

    context.user_data["func"] = i_func
    update.message.reply_text(
        'Insert starting point (eg: 1,0 )',
        reply_markup=ReplyKeyboardRemove()
    )

    return START


def n_iter(update: Update, context: CallbackContext) -> int:
    """Get the starting point"""

    #reply_keyboard = [['7','8','9'],['4','5','6'], [ '1', '2', '3'], ['0',',']]
    i_start = [int(x) for x in update.message.text.split(',')]
    context.user_data["start"] = i_start
    update.message.reply_text(
        'How many iteration or epsilon do you want?',
        reply_markup=ReplyKeyboardRemove()
    )

    return N_ITER


def result(update: Update, context: CallbackContext) -> int:
    """Get number of iterations"""
    user = update.message.from_user

    res = 0
    i_niter = int(update.message.text)
    context.user_data["niter"] = i_niter

    model = context.user_data["model"]
    obt = context.user_data["obt"]
    func = context.user_data["func"]
    start = context.user_data["start"]
    n_iter = context.user_data["niter"]

    if model == 'Gradient':
        res = do_grad(update, obt, func, start, n_iter)
    elif model == 'NewtonMult':
        res = do_newton(update, func, start, n_iter)
    elif model == 'Bisection':
        res = do_bisezione(update, obt, func, start, n_iter)
    elif model == 'NewtonUni':
        res = do_newton_uni(update, func, start[0], n_iter)
    # CALCULATE RESULT
    update.message.reply_text(
        'The result is ' + str(res)
    )

    return ConversationHandler.END


def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.

    # db = pickledb.load('example.db', False)# AUTO SAVE TO FILE = FALSE
    # db.set('key', 'value')
    # db.get('key')

    token = os.environ.get('TOKEN')
    updater = Updater(token)

    # We have to create a "symbol" called x
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', choose_opt)],
        states={
            MOD: [MessageHandler(Filters.regex('^(Gradient|NewtonMult|NewtonUni|Bisection)$'), obt)],
            OBT: [MessageHandler(Filters.regex('^(max|min)$'), func)],
            FUNC: [MessageHandler(Filters.text, start_point)],
            START: [MessageHandler(Filters.text, n_iter)],
            N_ITER: [MessageHandler(Filters.text, result)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    # updater.start_polling()
    updater.start_webhook(listen="0.0.0.0",
                          port=PORT,
                          url_path=token,
                          webhook_url='https://optimization-model.herokuapp.com/' + token)
    # updater.bot.set_webhook(url=settings.WEBHOOK_URL)
    #updater.bot.set_webhook('https://optimization-model.herokuapp.com/' + token)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


def do_grad(update, obt, func, x_k, n_iter):

    try:

        func = func.lower()
        if not (isinstance(func, Basic)):
            func = sympify(func)

        symbols = sorted(func.free_symbols, key=lambda symbol: symbol.name)

        if obt == 'min':
            sign = -1
        elif obt == 'max':
            sign = 1

        func_lam = lambdify(symbols, func)
        grad = [func.copy().diff(symbol) for symbol in symbols]
        print(bcolors.OKBLUE + 'gradiente: ' + bcolors.ENDC, grad)

        update.message.reply_text('gradiente: ' + str(grad))

        funcs = [lambdify(symbols, grad[i]) for i in range(len(symbols))]

        for iter in range(n_iter):

            print(bcolors.OKGREEN + '--------iter ' +
                  str(iter) + ' ---------' + bcolors.ENDC)
            update.message.reply_text('--------iter '+str(iter) + ' ---------')

            # ora calcolo grad_fs in start
            # questo e' il mio gradiente
            grad_fs = [funcs[i](*x_k) for i in range(len(x_k))]

            print(bcolors.OKBLUE + 'gradiente in start: ' + bcolors.ENDC, grad_fs)
            update.message.reply_text('gradiente in start: ' + str(grad_fs))

            #x_k_1 = x_k - alfak * grad_fs
            alfa = Symbol('a')
            mult = [Mul(x, alfa) for x in grad_fs]
            x_k_1 = [Add(x_k[i], sign * mult[i]) for i in range(len(mult))]

            print(bcolors.OKBLUE + 'xk1: ' + bcolors.ENDC, x_k_1)
            update.message.reply_text('xk1: ' + str(x_k_1))

            sub_f_xy = func_lam(*x_k_1)  # sostituisco nella func iniziale

            print(bcolors.OKBLUE + 'f(x,y) in xk1 :' + bcolors.ENDC, sub_f_xy)
            update.message.reply_text('f(x,y) in xk1 :' + str(sub_f_xy))

            der_f_xy = sub_f_xy.copy().diff(alfa)
            print(bcolors.OKBLUE + 'f\'(x,y) in xk1 :' + bcolors.ENDC, der_f_xy)
            update.message.reply_text('f\'(x,y) in xk1 :' + str(der_f_xy))

            res_alfa = solve(der_f_xy, alfa)

            print(bcolors.OKBLUE + 'alfa: ' + bcolors.ENDC, res_alfa)
            update.message.reply_text('alfa: ' + str(res_alfa))

            x_k_1_lam = lambdify(alfa, x_k_1)

            x_k = x_k_1_lam(*res_alfa)

        return x_k

    except:
        return ConversationHandler.END


def do_newton(update, func, x_k, n_iter):

    try:

        func = func.lower()
        if not (isinstance(func, Basic)):
            func = sympify(func)

        symbols = sorted(func.free_symbols, key=lambda symbol: symbol.name)
        func_lam = lambdify(symbols, func)
        grad = [func.copy().diff(symbol) for symbol in symbols]
        print(bcolors.OKBLUE + 'gradiente: ' + bcolors.ENDC, grad)
        update.message.reply_text('gradiente: ' + str(grad))
        funcs = [lambdify(symbols, grad[i]) for i in range(len(symbols))]
        hessiana = np.array([[grad[i].copy().diff(symbol) for symbol in symbols]
                             for i in range(len(grad))], dtype='float')
        #[[print(i, symbol) for symbol in symbols] for i in range(len(grad))]
        print(bcolors.OKBLUE + 'hessiana: \n' + bcolors.ENDC, hessiana)
        update.message.reply_text('hessiana: \n' + str(hessiana))
        hess_inv = np.linalg.inv(hessiana)
        print(bcolors.OKBLUE + 'hessiana -1 : \n' + bcolors.ENDC, hess_inv)
        update.message.reply_text('hessiana -1 : \n' + str(hess_inv))

        for iter in range(n_iter):
            update.message.reply_text('--------iter '+str(iter) + ' ---------')
            # ora calcolo grad_fs in start
            # questo e' il mio gradiente
            grad_fs = [funcs[i](*x_k) for i in range(len(x_k))]
            print(bcolors.OKBLUE + 'gradiente in start: ' + bcolors.ENDC, grad_fs)
            update.message.reply_text('gradiente in start: ' + str(grad_fs))

            print(bcolors.OKBLUE + 'FORMULA: x_k_'+str((iter+1))+' = \n' +
                  bcolors.ENDC+str(x_k) + ' - \n' + str(hess_inv) + ' * ' + str(grad_fs))
            update.message.reply_text('FORMULA: x_k_'+str((iter+1))+' = \n'+str(
                x_k) + ' - \n' + str(hess_inv) + ' * ' + str(grad_fs))
            x_k_1 = x_k - np.dot(hess_inv, grad_fs)

            print(bcolors.OKBLUE + 'x_k_'+str((iter+1)) +
                  ': ' + bcolors.ENDC, x_k_1)
            update.message.reply_text('x_k_'+str((iter+1))+': ' + str(x_k_1))

            x_k = x_k_1

        return x_k
    except:
        return ConversationHandler.END


def do_newton_uni(update, func, x_k, n_iter):

    try:
        func = func.lower()
        if not (isinstance(func, Basic)):
            func = sympify(func)

        symbol = func.free_symbols.pop()
        func_lam = lambdify(symbol, func)
        der_1 = func.copy().diff(symbol)
        der_2 = der_1.copy().diff(symbol)

        print(bcolors.OKBLUE + 'f\': ' + bcolors.ENDC, der_1)
        print(bcolors.OKBLUE + 'f": ' + bcolors.ENDC, der_2)

        update.message.reply_text('f\': ' + str(der_1))
        update.message.reply_text('f": ' + str(der_2))

        der_1_lam = lambdify(symbol, der_1)
        der_2_lam = lambdify(symbol, der_2)

        for iter in range(n_iter):
            update.message.reply_text('--------iter '+str(iter) + ' ---------')
            # ora calcolo grad_fs in start
            der_1_fs = der_1_lam(x_k)
            der_2_fs = der_2_lam(x_k)

            print(bcolors.OKBLUE + 'f\' in start: ' + bcolors.ENDC, der_1_fs)
            print(bcolors.OKBLUE + 'f" in start: ' + bcolors.ENDC, der_2_fs)
            update.message.reply_text('f\' in start: ' + str(der_1_fs))
            update.message.reply_text('f" in start: ' + str(der_2_fs))

            print(bcolors.OKBLUE + 'FORMULA: x_k_'+str((iter+1))+' = \n' +
                  bcolors.ENDC+str(x_k) + ' - \n' + str(der_1_fs) + ' / ' + str(der_2_fs))
            update.message.reply_text('FORMULA: x_k_'+str((iter+1))+' = \n'+str(
                x_k) + ' - \n' + str(der_1_fs) + ' / ' + str(der_2_fs))

            x_k_1 = x_k - (der_1_fs/der_2_fs)

            print(bcolors.OKBLUE + 'x_k_'+str((iter+1)) +
                  ': ' + bcolors.ENDC, x_k_1)
            update.message.reply_text('x_k_'+str((iter+1))+': ' + str(x_k_1))

            x_k = x_k_1

        return x_k
    except:
        return ConversationHandler.END


def do_bisezione(update, obt, func, x_k, epsilon):

    try:

        func = func.lower()
        if obt == 'min':
            sign = -1
        elif obt == 'max':
            sign = 1

        if not (isinstance(func, Basic)):
            func = sympify(func)

        symbols = sorted(func.free_symbols, key=lambda symbol: symbol.name)
        func_lam = lambdify(symbols, func)
        grad = [func.copy().diff(symbol) for symbol in symbols]
        print(bcolors.OKBLUE + 'gradiente: ' + bcolors.ENDC, grad)
        update.message.reply_text('gradiente: ' + str(grad))
        funcs = [lambdify(symbols, grad[i]) for i in range(len(symbols))]
        xs = [func(v) for v in x_k for func in funcs]
        print(xs)
        update.message.reply_text(xs)

        mids = []

        # controllo se sono discordi

        a = 1
        for i in xs:
            a = a * i

        if a < 0:
            # if ((xs[0] > 0) and (xs[1] < 0)) or ((xs[0] < 0) and (xs[1] > 0)):

            print(bcolors.FAIL + 'sono discordi' + bcolors.ENDC)
            while True:
                print(bcolors.OKBLUE + 'il minimo e\' compreso in: ' +
                      bcolors.ENDC, x_k)
                update.message.reply_text(
                    'il minimo e\' compreso in: ' + str(x_k))
                x_neg = min(x_k)
                x_pos = max(x_k)

                # prendo valore medio
                mid = np.mean(x_k)
                mids.append(mid)
                print(bcolors.OKBLUE + 'il minimo e\' : ' + bcolors.ENDC, mid)
                update.message.reply_text('il minimo e\' : ' + str(mid))

                new_x = funcs[0](mid)

                if obt == 'min':
                    if new_x > 0:
                        # prende il posto di quello positivo
                        x_pos = mid
                    else:
                        # prende il posto di quello negativo
                        x_neg = mid
                else:  # max
                    if new_x < 0:
                        # prende il posto di quello positivo
                        x_pos = mid
                    else:
                        # prende il posto di quello negativo
                        x_neg = mid

                x_k = (x_neg, x_pos)

                print(bcolors.OKBLUE + 'la f nel minimo vale: ' +
                      bcolors.ENDC, func_lam(mid))
                update.message.reply_text(
                    'la f nel minimo vale: ' + str(func_lam(mid)))

                if len(mids) > 1:
                    print(
                        'abs(' + str(func_lam(mids[-1])) + '-' + str(func_lam(mids[-2])) + ' < ' + str(epsilon))
                    update.message.reply_text(
                        'abs(' + str(func_lam(mids[-1])) + '-' + str(func_lam(mids[-2])) + ' < ' + str(epsilon))
                    if (abs(func_lam(mids[-1]) - func_lam(mids[-2])) < epsilon):

                        # ho finito
                        print(bcolors.FAIL + 'finito' + bcolors.ENDC)
                        return func_lam(mid)

        else:
            print(bcolors.FAIL + 'sono concordi' + bcolors.ENDC)
            return -1
            # altrimenti prendo nuovo valore e ricalcolo media
    except:
        return ConversationHandler.END


if __name__ == '__main__':
    main()
